"""
Batch plane-strain finite element analysis of kidney sections with PyMAPDL.

For every case it reads the DXF (and CSV landmarks) written by extract_kidney_outlines.py,
builds a 2D plane-strain model in MAPDL, applies the same material, mesh rules, boundary
conditions and intrapelvic pressure to every section, solves with large deformation, and
writes standardized, pressure-normalized outputs.

Requirements:
    pip install ansys-mapdl-core ezdxf numpy scipy matplotlib
    plus a licensed Ansys Mechanical APDL installation (local, or OSC: `module load ansys`)

Usage:
    python batch_fea.py outlines/UUO_POD3 outlines/UUO_POD7 outlines/UUO_POD1 --out fea
    python batch_fea.py outlines/UUO_POD3 --mesh-scale 0.5 --tag fine     # mesh check
    python batch_fea.py outlines/UUO_POD3 --dry-run                       # APDL + geometry check only

Each case is given by its base path (without extension); <base>.dxf and <base>.csv must
exist. A DXF re-exported from CAD (e.g. Fusion) without the KIDNEY_OUTER / PELVIS_LUMEN
layers is accepted: the kidney outline is taken as the largest closed loop and the lumen as
the largest loop inside it. Do not move or scale the sketch, so the landmarks in the CSV
still match; check <case>_geometry_check.png from --dry-run. Units: mm, N, MPa (pressure 200 Pa = 0.0002 MPa).

Boundary sizing: the lumen edge length follows the local radius of curvature (0.3 R),
between 2 um (curves, slot ends, and within --fornix-r of the fornices) and 20 um on
straight parts, with at most 30% change between neighbours. --mesh-scale scales every
size for the mesh-convergence study.

Model (identical for all cases):
    PLANE183, plane strain (KEYOPT(3)=2), mixed u-P (KEYOPT(6)=1), triangles
    Neo-Hookean: W = mu/2 (I1bar - 3) + (1/d)(J - 1)^2, mu = E/(2(1+nu)), d = 2/K,
                 K = E/(3(1-2 nu))
    Uniform pressure on all pelvic-lumen edges; minimal rigid-body constraints on the
    outer boundary: leftmost node UX=UY=0, rightmost node UY=0
    NLGEOM on, automatic time stepping

Contour maps (matplotlib, identical styling for every case):
    <case>_<field>_full.png and <case>_<field>_zoom.png for von Mises / p, maximum principal
    strain, mean stress / p and displacement. Use --clim to share colour limits across cases
    (needed for comparing time points), --deformed for the deformed shape, and --replot to
    redraw from the saved <case>_fields.npz without re-running MAPDL.

Outputs in --out:
    summary.csv                      one row per case (appended)
    <case>_boundary.csv              strain and displacement along the lumen boundary
    <case>.inp                       the APDL commands used (for the record)
"""
import argparse, csv, os, sys
import numpy as np
import ezdxf
from scipy.spatial import ConvexHull
from matplotlib.path import Path


# ---------------------------------------------------------------- geometry ----
def entity_points(e, h):
    """Sample a DXF entity at roughly spacing h (mm)."""
    t = e.dxftype()
    if t == "SPLINE":
        pts = np.array([(p.x, p.y) for p in e.flattening(h / 10)])
    elif t == "ARC":
        pts = np.array([(p.x, p.y) for p in e.flattening(h / 50)])
    elif t == "LWPOLYLINE":
        pts = np.array([p[:2] for p in e.get_points()])
        if e.closed:
            pts = np.vstack([pts, pts[:1]])
    elif t == "LINE":
        pts = np.array([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)])
    else:
        return None
    return pts


def chain(segments, tol=1e-6):
    """Join open polylines end to end into one closed loop."""
    segs = [s for s in segments if s is not None and len(s) > 1]
    loop = segs.pop(0)
    while segs:
        end = loop[-1]
        best, flip, dmin = None, False, np.inf
        for i, s in enumerate(segs):
            for f, p in ((False, s[0]), (True, s[-1])):
                d = np.hypot(*(p - end))
                if d < dmin:
                    best, flip, dmin = i, f, d
        if dmin > 1e-3:
            raise ValueError(f"Lumen pieces do not join (gap {dmin:.2e} mm)")
        s = segs.pop(best)
        s = s[::-1] if flip else s
        loop = np.vstack([loop, s[1:]])
    if np.hypot(*(loop[0] - loop[-1])) < 1e-3:
        loop = loop[:-1]
    return loop


def resample(loop, h):
    """Resample a closed loop at uniform arc-length spacing ~h."""
    closed = np.vstack([loop, loop[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    n = max(int(np.ceil(s[-1] / h)), 12)
    si = np.linspace(0, s[-1], n, endpoint=False)
    return np.c_[np.interp(si, s, closed[:, 0]), np.interp(si, s, closed[:, 1])]


def rdp(pts, eps):
    """Douglas-Peucker simplification of an open polyline (iterative)."""
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        v = b - a
        L = np.hypot(*v)
        q = pts[i + 1:j] - a
        d = np.abs(v[0] * q[:, 1] - v[1] * q[:, 0]) / L if L > 0 else np.hypot(*q.T)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def adaptive(loop, h):
    """Closed loop from a curvature-adaptive flattening: drop points closer than h/4 and
    subdivide segments longer than h, so tight curves (slot ends, fornices) stay resolved."""
    keep = [loop[0]]
    for p in loop[1:]:
        if np.hypot(*(p - keep[-1])) >= h / 4:
            keep.append(p)
    if np.hypot(*(keep[0] - keep[-1])) < h / 4:
        keep.pop()
    keep = np.array(keep)
    # drop points that add no shape (Douglas-Peucker, tolerance h/40) before subdividing
    i0 = 0
    i1 = int(np.argmax(np.hypot(*(keep - keep[0]).T)))
    keep = np.vstack([rdp(keep[i0:i1 + 1], h / 40)[:-1], rdp(np.vstack([keep[i1:], keep[:1]]), h / 40)[:-1]])
    out = []
    for p, q in zip(keep, np.roll(keep, -1, axis=0)):
        k = max(int(np.ceil(np.hypot(*(q - p)) / h)), 1)
        out.extend(p + (q - p) * t for t in np.arange(k) / k)
    return np.array(out)


def resample_open(pts, h):
    """Resample an open polyline at spacing ~h, keeping both end points."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    n = max(int(np.ceil(s[-1] / h)), 2)
    si = np.linspace(0, s[-1], n + 1)
    return np.c_[np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])]


def curvature_size(loop, hmin, hmax, k, fine_pts=(), fine_r=0.0, grow=1.3):
    """Re-space a closed loop so the edge length follows the local radius of curvature:
    about k * R, clipped to [hmin, hmax], hmin within fine_r of the fine_pts (fornices),
    and changing by at most a factor `grow` between neighbours."""
    dense = adaptive(loop, hmin)
    n = len(dense)
    # radius of curvature from a three-point circle over a window of ~3*hmin on each side
    w = max(int(round(3 * hmin / max(np.median(np.hypot(*np.diff(dense, axis=0).T)), 1e-9))), 1)
    a, b, c = np.roll(dense, w, axis=0), dense, np.roll(dense, -w, axis=0)
    ab, bc, ca = (np.hypot(*(b - a).T), np.hypot(*(c - b).T), np.hypot(*(a - c).T))
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    R = np.where(cross > 1e-15, ab * bc * ca / (2 * np.maximum(cross, 1e-15)), np.inf)
    t = np.clip(k * R, hmin, hmax)
    for q in fine_pts:
        t[np.hypot(*(dense - q).T) <= fine_r] = hmin
    for _ in range(2):                     # limit size jumps in both directions, around the loop
        for i in range(1, 2 * n):
            j, p = i % n, (i - 1) % n
            t[j] = min(t[j], t[p] * grow)
        for i in range(2 * n - 1, 0, -1):
            j, p = (i - 1) % n, i % n
            t[j] = min(t[j], t[p] * grow)
    seg = np.hypot(*np.diff(np.vstack([dense, dense[:1]]), axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    tt = np.r_[t, t[:1]]
    out, pos = [], 0.0
    while pos < s[-1] - 0.5 * np.interp(pos, s, tt):
        out.append(pos)
        pos += np.interp(pos, s, tt)
    closed = np.vstack([dense, dense[:1]])
    out = np.array(out)
    return np.c_[np.interp(out, s, closed[:, 0]), np.interp(out, s, closed[:, 1])]


def loops_from_pieces(pieces, tol=1e-3):
    """Group open curve pieces into closed loops by joining matching end points."""
    pieces = [p for p in pieces if p is not None and len(p) > 1]
    loops = []
    while pieces:
        loop = pieces.pop(0)
        grew = True
        while grew and np.hypot(*(loop[0] - loop[-1])) > tol:
            grew = False
            for i, p in enumerate(pieces):
                for cand in (p, p[::-1]):
                    if np.hypot(*(cand[0] - loop[-1])) <= tol:
                        loop = np.vstack([loop, cand[1:]])
                        pieces.pop(i)
                        grew = True
                        break
                if grew:
                    break
        if np.hypot(*(loop[0] - loop[-1])) <= tol and len(loop) > 3:
            loops.append(loop[:-1])
    return loops


def unlabelled_outlines(msp, h):
    """For a DXF re-exported from CAD (layer names lost): the kidney outline is the largest
    closed loop that is not a rectangle (e.g. the canvas frame); the lumen is the largest
    loop inside it."""
    from matplotlib.path import Path as _Path
    pieces = []
    for e in msp:
        if e.dxftype() in ("SPLINE", "ARC", "LINE", "LWPOLYLINE"):
            pieces.append(entity_points(e, h))
        elif e.dxftype() == "POLYLINE":
            pts = np.array([(v.dxf.location.x, v.dxf.location.y) for v in e.vertices])
            pieces.append(np.vstack([pts, pts[:1]]) if e.is_closed else pts)
        elif e.dxftype() == "CIRCLE":
            c, r = e.dxf.center, e.dxf.radius
            t = np.linspace(0, 2 * np.pi, 64)
            pieces.append(np.c_[c.x + r * np.cos(t), c.y + r * np.sin(t)])
    loops = loops_from_pieces(pieces)
    def is_rect(p):
        w, hh = np.ptp(p[:, 0]), np.ptp(p[:, 1])
        return w * hh > 0 and poly_area(p) > 0.98 * w * hh
    loops = sorted([p for p in loops if not is_rect(p)], key=poly_area, reverse=True)
    if len(loops) < 2:
        raise ValueError("could not find a kidney outline and a lumen loop in the DXF")
    outer = loops[0]
    inside = [p for p in loops[1:] if _Path(outer).contains_points(p).all()]
    if not inside:
        raise ValueError("no closed lumen loop found inside the kidney outline")
    w = np.ptp(outer[:, 0])
    if not 0.5 < w < 50:
        print(f"  warning: kidney outline is {w:.3g} units wide; the DXF should be in mm")
    print("  note: DXF has no KIDNEY_OUTER / PELVIS_LUMEN layers; outlines identified by size")
    return outer, inside[0]


def read_case(base, outer_h, lumen_h, hmax=None, k=0.3, fine_r=0.0):
    doc = ezdxf.readfile(base + ".dxf")
    msp = doc.modelspace()
    outer = [entity_points(e, outer_h) for e in msp if e.dxf.layer == "KIDNEY_OUTER"]
    lum_e = [e for e in msp if e.dxf.layer == "PELVIS_LUMEN"]
    if not outer or not lum_e:   # re-exported from CAD: identify the loops geometrically
        o_loop, l_loop = unlabelled_outlines(msp, lumen_h)
        outer = resample(o_loop, outer_h)
        lumen = adaptive(np.vstack([l_loop, l_loop[:1]]), lumen_h)
        lum_e = None
    else:
        outer = resample(chain(outer), outer_h)
    if lum_e is None:
        pass
    elif len(lum_e) == 1:      # one closed curve (automatic or hybrid): curvature-adaptive
        lumen = adaptive(chain([entity_points(lum_e[0], lumen_h)]), lumen_h)
    else:                    # slot: two splines + two end caps; caps get >= 12 segments
        pieces = []
        for e in lum_e:
            pts = entity_points(e, lumen_h)
            h = min(lumen_h, np.pi * e.dxf.radius / 12) if e.dxftype() == "ARC" else lumen_h
            pieces.append(resample_open(pts, h))
        lumen = chain(pieces)
    marks = {}
    if os.path.exists(base + ".csv"):
        with open(base + ".csv") as f:
            for row in csv.DictReader(f):
                if row["contour"] in ("PAPILLA_TIP", "FORNIX_L", "FORNIX_R"):
                    marks[row["contour"]] = np.array([float(row["x_mm"]), float(row["y_mm"])])
    if hmax and hmax > lumen_h:   # coarser where the boundary is straight, fine at curves/fornices
        fine = [marks[k_] for k_ in ("FORNIX_L", "FORNIX_R") if k_ in marks]
        lumen = curvature_size(lumen, lumen_h, hmax, k, fine, fine_r)
    return outer, lumen, marks


def split_sides(lumen, marks):
    """Split the lumen loop at the two fornices into the papillary surface and the pelvic
    wall. Returns a boolean mask
    (True = papilla side) and the papilla polygon (papillary arc closed between fornices)."""
    n = len(lumen)
    if not all(k in marks for k in ("FORNIX_L", "FORNIX_R")):
        hull = lumen[ConvexHull(lumen).vertices]
        return Path(hull).contains_points(lumen) & (dist_to_polygon(lumen, hull) > 0.004), hull
    near = lambda q: int(np.argmin(np.hypot(*(lumen - q).T)))
    iL, iR = near(marks["FORNIX_L"]), near(marks["FORNIX_R"])
    arc1 = [(iL + k) % n for k in range((iR - iL) % n + 1)]
    arc2 = [(iR + k) % n for k in range((iL - iR) % n + 1)]
    # each arc closed by the fornix-to-fornix chord: the papillary arc encloses only the
    # papilla, the wall arc encloses the papilla plus the pelvic space -> pick the smaller
    pap = arc1 if poly_area(lumen[arc1]) < poly_area(lumen[arc2]) else arc2
    mask = np.zeros(n, bool)
    mask[pap] = True
    mask[[iL, iR]] = False
    return mask, lumen[pap]


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def dist_to_polygon(pts, poly):
    a, b = poly, np.roll(poly, -1, axis=0)
    d = np.full(len(pts), np.inf)
    for p0, p1 in zip(a, b):
        v = p1 - p0
        t = np.clip(((pts - p0) @ v) / (v @ v), 0, 1)
        d = np.minimum(d, np.hypot(*(pts - (p0 + t[:, None] * v)).T))
    return d


# ---------------------------------------------------------------- APDL --------
def apdl_commands(outer, lumen, a):
    mu = a.E / (2 * (1 + a.nu))
    K = a.E / (3 * (1 - 2 * a.nu))
    d = 2 / K
    no, nl = len(outer), len(lumen)
    c = ["/PREP7", "ET,1,PLANE183", "KEYOPT,1,3,2", "KEYOPT,1,6,1",
         "TB,HYPER,1,,,NEO", f"TBDATA,1,{mu:.6e},{d:.6e}"]
    for i, (x, y) in enumerate(outer, 1):
        c.append(f"K,{i},{x:.7f},{y:.7f}")
    for i, (x, y) in enumerate(lumen, no + 1):
        c.append(f"K,{i},{x:.7f},{y:.7f}")
    for i in range(no):
        c.append(f"L,{i + 1},{(i + 1) % no + 1}")
    for i in range(nl):
        c.append(f"L,{no + i + 1},{no + (i + 1) % nl + 1}")
    c += [f"LSEL,S,LINE,,1,{no}", "AL,ALL",
          f"LSEL,S,LINE,,{no + 1},{no + nl}", "AL,ALL",
          "ALLSEL", "ASBA,1,2",
          f"LSEL,S,LINE,,1,{no}", "LESIZE,ALL,,,1",
          f"LSEL,S,LINE,,{no + 1},{no + nl}", "LESIZE,ALL,,,1",
          "ALLSEL", f"ESIZE,{a.esize}", "MSHAPE,1,2D", "MSHKEY,0", "MOPT,TRANS,1.5",
          "AMESH,ALL", "FINISH"]
    return c, no, nl


# ---------------------------------------------------------------- contour plots -----
FIELDS = {   # key: (label, colormap, default limits or None for automatic)
    "seqv_over_p": ("von Mises stress / pressure", "viridis", None),
    "e1": ("maximum principal strain", "magma", None),
    "hydro_over_p": ("mean (hydrostatic) stress / pressure  (< 0: compression)", "RdBu_r", None),
    "disp_um": ("displacement magnitude (um)", "cividis", None),
}


def parse_clim(text):
    """'seqv_over_p=0,10;e1=0,0.05' -> {'seqv_over_p': (0, 10), 'e1': (0, 0.05)}"""
    out = {}
    for part in (text or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            lo, hi = (float(x) for x in v.split(","))
            out[k.strip()] = (lo, hi)
    return out


def plot_case(npz_path, a):
    """Contour maps of one case from its saved field file, full section and pelvis zoom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    d = np.load(npz_path, allow_pickle=True)
    name = str(d["name"])
    xy = d["xy"] + (d["u"] if a.deformed else 0.0)
    tri = Triangulation(xy[:, 0], xy[:, 1], d["tri"])
    lumen, outer = d["lumen"], d["outer"]
    clim = parse_clim(a.clim)
    lo_, hi_ = lumen.min(axis=0), lumen.max(axis=0)
    pad = max(0.25, 0.3 * float(np.max(hi_ - lo_)))
    views = {"full": None, "zoom": (lo_[0] - pad, hi_[0] + pad, lo_[1] - pad, hi_[1] + pad)}
    files = []
    for key in a.fields.split(","):
        key = key.strip()
        if key not in FIELDS:
            print(f"  unknown field {key}; choose from {', '.join(FIELDS)}")
            continue
        label, cmap, lim = FIELDS[key]
        v = d[key]
        lim = clim.get(key, lim)
        if lim is None:   # robust automatic limits (ignore the extreme 0.5% at singular points)
            if key == "hydro_over_p":
                m_ = float(np.percentile(np.abs(v), 99.5))
                lim = (-m_, m_)
            else:
                lim = (float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5)))
        for view, box in views.items():
            fig, ax = plt.subplots(figsize=(6.5, 6))
            pc = ax.tripcolor(tri, v, shading="gouraud", cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.plot(*np.vstack([outer, outer[:1]]).T, "k-", lw=0.6)
            ax.plot(*np.vstack([lumen, lumen[:1]]).T, "w-", lw=0.6)
            for k_ in ("FORNIX_L", "FORNIX_R", "PAPILLA_TIP"):
                if k_ in d.files and d[k_].size == 2:
                    ax.plot(*d[k_], "o", ms=3, mfc="none", mec="w")
            if box:
                ax.set_xlim(box[0], box[1])
                ax.set_ylim(box[2], box[3])
            ax.set_aspect("equal")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            p_pa = a.title_pressure_pa if a.title_pressure_pa else float(d["pressure_MPa"]) * 1e6
            shape = "deformed" if a.deformed else "undeformed"
            ax.set_title(f"{name}  |  p = {p_pa:.0f} Pa  |  {shape}", fontsize=9)
            over, under = bool(np.nanmax(v) > lim[1]), bool(np.nanmin(v) < lim[0])
            ext = "both" if over and under else "max" if over else "min" if under else "neither"
            cb = fig.colorbar(pc, ax=ax, shrink=0.8, extend=ext)   # arrows: values beyond the scale
            cb.set_label(label, fontsize=8)
            fig.tight_layout()
            for ext in a.fmt.split(","):
                f = os.path.join(os.path.dirname(npz_path), f"{name}_{key}_{view}.{ext.strip()}")
                fig.savefig(f, dpi=a.dpi)
                files.append(f)
            plt.close(fig)
    print(f"  saved {len(files)} contour plots for {name}")
    return files


# ---------------------------------------------------------------- run ---------
def run_case(mapdl, base, a, writer):
    name = os.path.basename(base) + (f"_{a.tag}" if a.tag else "")
    ms = a.mesh_scale
    a.lumen_h, a.outer_h, a.esize, a.lumen_hmax = (a.lumen_h0 * ms, a.outer_h0 * ms,
                                                    a.esize0 * ms, a.lumen_hmax0 * ms)
    outer, lumen, marks = read_case(base, a.outer_h, a.lumen_h, a.lumen_hmax, a.curv_k, a.fornix_r)
    cmds, no, nl = apdl_commands(outer, lumen, a)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, name + ".inp"), "w") as f:
        f.write("\n".join(cmds) + "\n")
    print(f"{name}: {no} outer + {nl} lumen boundary points")
    # clearance between the pelvic space and the outer surface: a very thin wall (or a lumen
    # crossing the outline) makes the model ill-posed and usually stops convergence
    from matplotlib.path import Path as _P
    if not _P(outer).contains_points(lumen).all():
        print("  WARNING: part of the lumen lies OUTSIDE the kidney outline - fix the geometry")
    else:
        dmin = float(dist_to_polygon(lumen, outer).min())
        j = int(np.argmin(dist_to_polygon(lumen, outer)))
        msg = f"  minimum wall between lumen and outer surface: {dmin * 1000:.0f} um at ({lumen[j, 0]:.2f}, {lumen[j, 1]:.2f}) mm"
        print(msg + ("   <-- too thin, aim for >= 50-100 um" if dmin < 0.05 else ""))
    try:   # geometry check figure: outlines, landmarks, papilla / wall split
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        side, pap = split_sides(lumen, marks)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(*np.vstack([outer, outer[:1]]).T, "k-", lw=0.7)
        ax.fill(*pap.T, color="orange", alpha=0.3, label="papilla region")
        ax.plot(*lumen[side].T, "r.", ms=1.5, label="papilla side")
        ax.plot(*lumen[~side].T, "b.", ms=1.5, label="wall side")
        for k_, v_ in marks.items():
            ax.plot(*v_, "ko", ms=4)
            ax.annotate(k_, v_, fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"{name}: geometry check (mm)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, name + "_geometry_check.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"  geometry check figure skipped: {exc}")
    if a.dry_run:
        return

    import time
    t0 = time.time()
    stamp = lambda msg: print(f"  [{time.time() - t0:6.0f} s] {msg}", flush=True)
    stamp("building geometry and mesh in MAPDL ...")
    mapdl.clear()
    mapdl.input_strings("\n".join(cmds)) if hasattr(mapdl, "input_strings") else None
    if not hasattr(mapdl, "input_strings"):
        with mapdl.non_interactive:
            for cmd in cmds:
                mapdl.run(cmd)
    stamp("mesh done")
    mapdl.prep7()
    mapdl.allsel()
    nnum = mapdl.mesh.nnum
    xy = mapdl.mesh.nodes[:, :2]
    nnum0, xy0 = np.array(nnum), np.array(xy)
    print(f"  {mapdl.mesh.n_elem} elements, {len(nnum)} nodes")

    # boundary-condition nodes: leftmost and rightmost nodes of the outer boundary
    mapdl.lsel("S", "LINE", "", 1, no)
    mapdl.nsll("S", 1)
    on = np.asarray(mapdl.get_array("NODE", item1="NLIST"), int)
    pos = {n: i for i, n in enumerate(nnum)}
    oxy = xy[[pos[n] for n in on]]
    nA, nB = int(on[np.argmin(oxy[:, 0])]), int(on[np.argmax(oxy[:, 0])])
    # lumen boundary nodes at the keypoints, in loop order
    lum_nodes = []
    for (x, y) in lumen:
        lum_nodes.append(int(nnum[np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y))]))
    mapdl.allsel()

    mapdl.slashsolu()
    mapdl.antype("STATIC")
    mapdl.nlgeom("ON")
    mapdl.autots("ON")
    mapdl.nsubst(a.nsub, 1000, max(a.nsub // 2, 1))
    mapdl.outres("ALL", "ALL")
    mapdl.d(nA, "UX", 0)
    mapdl.d(nA, "UY", 0)
    mapdl.d(nB, "UY", 0)
    mapdl.lsel("S", "LINE", "", no + 1, no + nl)
    mapdl.sfl("ALL", "PRES", a.pressure)
    mapdl.allsel()
    converged = True
    stamp(f"solving (large deformation, {a.nsub} initial substeps) ...")
    try:
        mapdl.solve()
    except Exception as exc:
        converged = False
        print(f"  solve did not converge: {str(exc).splitlines()[0][:120]}")
    mapdl.finish()
    stamp("solve finished; post-processing ...")

    mapdl.post1()
    mapdl.set("LAST")
    load = float(mapdl.post_processing.time)
    if not converged or mapdl.post_processing.sub_step >= 999999:
        # the last set is MAPDL's unconverged debug output: use the last converged one
        mapdl.set("PREVIOUS")
        load = float(mapdl.post_processing.time)
        converged = False
    p_eff = a.pressure * min(load, 1.0)
    mapdl.allsel()
    pp = mapdl.post_processing
    seqv = pp.nodal_eqv_stress()
    # node numbers/coordinates matching the result arrays: after solving, the mesh query can
    # also list solver-internal nodes, so use the PREP7 node list when the sizes differ
    nn, xy = mapdl.mesh.nnum, mapdl.mesh.nodes[:, :2]
    if len(nn) != len(seqv) or len(xy) != len(seqv):
        if len(nnum0) == len(seqv):
            nn, xy = nnum0, xy0
        else:
            raise RuntimeError(f"result size {len(seqv)} does not match the mesh "
                               f"({len(nnum0)} nodes before solving, {len(nn)} after)")
    sx, sy, sz = (pp.nodal_component_stress(k) for k in ("X", "Y", "Z"))
    ex, ey, gxy = (pp.nodal_elastic_component_strain(k) for k in ("X", "Y", "XY"))
    ux, uy = pp.nodal_displacement("X"), pp.nodal_displacement("Y")
    e1 = (ex + ey) / 2 + np.sqrt(((ex - ey) / 2) ** 2 + (gxy / 2) ** 2)
    hydro = (sx + sy + sz) / 3            # mean stress; negative = compression
    idx = {n: i for i, n in enumerate(nn)}

    # fornix stress concentration: max von Mises within fornix_r of each fornix
    row = {"case": name, "mesh_scale": a.mesh_scale, "pressure_MPa": a.pressure, "load_factor": round(min(load, 1.0), 4),
           "converged": converged and load >= 1.0 - 1e-6, "E_MPa": a.E, "nu": a.nu,
           "lumen_h_mm": a.lumen_h, "n_elem": mapdl.mesh.n_elem}
    for k in ("FORNIX_L", "FORNIX_R"):
        if k in marks:
            m = np.hypot(*(xy - marks[k]).T) <= a.fornix_r
            row[f"SCF_{k}"] = round(float(seqv[m].max() / p_eff), 3) if m.any() else None
    row["SCF_global"] = round(float(seqv.max() / p_eff), 3)

    # element triangles (corner nodes), used for area-weighted averages and the contour maps
    tri = []
    try:   # element corner nodes (MAPDL element record: 10 attributes, then the nodes)
        for e in mapdl.mesh.elem:
            corners = [int(n) for n in e[10:13]]
            if len(set(corners)) == 3 and all(n in idx for n in corners):
                tri.append([idx[n] for n in corners])
    except Exception:
        tri = []
    if len(tri) < mapdl.mesh.n_elem // 2:   # fallback: triangulate the nodes, keep tissue only
        from scipy.spatial import Delaunay
        t_ = Delaunay(xy).simplices
        cen = xy[t_].mean(axis=1)
        keep = Path(outer).contains_points(cen) & ~Path(lumen).contains_points(cen)
        tri = t_[keep].tolist()
        print("  note: element connectivity not read; contour plots use a node triangulation")
    tri = np.array(tri, int)
    tri_xy = xy[tri]
    tri_area = 0.5 * np.abs((tri_xy[:, 1, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 2, 1] - tri_xy[:, 0, 1]) -
                            (tri_xy[:, 2, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 1, 1] - tri_xy[:, 0, 1]))
    tri_cen = tri_xy.mean(axis=1)
    for k in ("FORNIX_L", "FORNIX_R"):
        if k in marks:
            v_ = fornix_mean_scf(xy, tri, seqv / p_eff, marks[k], a.fornix_mean_r)
            row[f"SCFmean_{k}"] = round(v_, 3) if v_ is not None else None

    # papilla: tissue enclosed by the papillary surface, closed between the two fornices.
    # Area-weighted mean (a plain node average would be biased toward the fine mesh at the
    # fornices); the median is reported as well.
    papilla_side, pap_poly = split_sides(lumen, marks)
    in_t = Path(pap_poly).contains_points(tri_cen)
    if in_t.any():
        hv = hydro[tri[in_t]].mean(axis=1) / p_eff
        w_ = tri_area[in_t]
        row["papilla_hydro_over_p"] = round(float(np.sum(hv * w_) / np.sum(w_)), 3)
        order = np.argsort(hv)
        cw = np.cumsum(w_[order]) / np.sum(w_)
        row["papilla_hydro_median_over_p"] = round(float(hv[order][np.searchsorted(cw, 0.5)]), 3)
        row["papilla_area_mm2"] = round(float(np.sum(w_)), 4)
    if "PAPILLA_TIP" in marks:
        j = np.argmin(np.hypot(*(xy - marks["PAPILLA_TIP"]).T))
        row["tip_disp_um"] = round(float(np.hypot(ux[j], uy[j]) * 1000), 3)

    # lumen boundary: area change and surface strain, split into papilla / wall side
    # (the boundary is split at the two fornices; see split_sides)
    li = np.array([idx[n] for n in lum_nodes])
    def_xy = lumen + np.c_[ux[li], uy[li]]
    A0, A1 = poly_area(lumen), poly_area(def_xy)
    row["lumen_area0_mm2"] = round(A0, 5)
    row["dA_over_A0_per_kPa"] = round((A1 - A0) / A0 / (p_eff * 1000), 4)
    # surface strain per side: the maximum (near the fornices it is mesh dependent) plus the
    # median and 95th percentile excluding the fornix regions (more robust for comparisons)
    away = np.ones(len(lumen), bool)
    for k_ in ("FORNIX_L", "FORNIX_R"):
        if k_ in marks:
            away &= np.hypot(*(lumen - marks[k_]).T) > a.fornix_r
    for side, msk in (("papilla", papilla_side), ("wall", ~papilla_side)):
        row[f"max_e1_{side}"] = round(float(e1[li][msk].max()), 5) if msk.any() else None
        mm_ = msk & away
        if mm_.any():
            row[f"median_e1_{side}"] = round(float(np.median(e1[li][mm_])), 5)
            row[f"p95_e1_{side}"] = round(float(np.percentile(e1[li][mm_], 95)), 5)
    writer.writerow(row)

    # field file for contour plots (and later re-plotting with --replot)
    try:
        u_xy = np.c_[ux, uy]
        npz = os.path.join(a.out, name + "_fields.npz")
        np.savez_compressed(npz, name=name, xy=xy, u=u_xy, tri=tri,
                            seqv_over_p=seqv / p_eff, e1=e1, hydro_over_p=hydro / p_eff,
                            disp_um=np.hypot(ux, uy) * 1000, lumen=lumen, outer=outer,
                            pressure_MPa=p_eff, **{k: v for k, v in marks.items()})
        if not a.no_plots:
            plot_case(npz, a)
    except Exception as exc:
        print(f"  contour plots skipped: {exc}")

    s = np.r_[0, np.cumsum(np.hypot(*np.diff(lumen, axis=0).T))]
    with open(os.path.join(a.out, name + "_boundary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s_mm", "x_mm", "y_mm", "side", "e1", "seqv_over_p", "ux_um", "uy_um"])
        for k, i in enumerate(li):
            w.writerow([round(s[k], 5), round(lumen[k, 0], 5), round(lumen[k, 1], 5),
                        "papilla" if papilla_side[k] else "wall", round(float(e1[i]), 6),
                        round(float(seqv[i] / p_eff), 4), round(float(ux[i]) * 1000, 4),
                        round(float(uy[i]) * 1000, 4)])
    print(f"  done: load factor {row['load_factor']}, SCF_global {row['SCF_global']}")


def fornix_mean_scf(xy, tri, seqv_over_p, mark, r):
    """Area-weighted mean von Mises / p over the tissue within radius r of a fornix
    (mesh-independent alternative to the singular peak value)."""
    txy = xy[tri]
    area = 0.5 * np.abs((txy[:, 1, 0] - txy[:, 0, 0]) * (txy[:, 2, 1] - txy[:, 0, 1]) -
                        (txy[:, 2, 0] - txy[:, 0, 0]) * (txy[:, 1, 1] - txy[:, 0, 1]))
    inn = np.hypot(*(txy.mean(axis=1) - mark).T) <= r
    if not inn.any():
        return None
    v = seqv_over_p[tri[inn]].mean(axis=1)
    return float(np.sum(v * area[inn]) / np.sum(area[inn]))


def rescore(npz_path, a):
    """Recompute the papilla and boundary metrics from a saved field file (no MAPDL)."""
    d = np.load(npz_path, allow_pickle=True)
    xy, tri = d["xy"], d["tri"]
    lumen = d["lumen"]
    marks = {k: d[k] for k in ("PAPILLA_TIP", "FORNIX_L", "FORNIX_R") if k in d.files}
    hydro, e1 = d["hydro_over_p"], d["e1"]
    row = {"case": str(d["name"])}
    txy = xy[tri]
    area = 0.5 * np.abs((txy[:, 1, 0] - txy[:, 0, 0]) * (txy[:, 2, 1] - txy[:, 0, 1]) -
                        (txy[:, 2, 0] - txy[:, 0, 0]) * (txy[:, 1, 1] - txy[:, 0, 1]))
    side, pap = split_sides(lumen, marks)
    inn = Path(pap).contains_points(txy.mean(axis=1))
    hv, w_ = hydro[tri[inn]].mean(axis=1), area[inn]
    row["papilla_hydro_over_p"] = round(float(np.sum(hv * w_) / np.sum(w_)), 3)
    for k_ in ("FORNIX_L", "FORNIX_R"):
        if k_ in marks:
            v_ = fornix_mean_scf(xy, tri, d["seqv_over_p"], marks[k_], a.fornix_mean_r)
            row[f"SCFmean_{k_}"] = round(v_, 3) if v_ is not None else None
    o_ = np.argsort(hv)
    row["papilla_hydro_median_over_p"] = round(float(hv[o_][np.searchsorted(np.cumsum(w_[o_]) / np.sum(w_), 0.5)]), 3)
    from scipy.spatial import cKDTree
    li = cKDTree(xy).query(lumen)[1]
    away = np.ones(len(lumen), bool)
    for k_ in ("FORNIX_L", "FORNIX_R"):
        if k_ in marks:
            away &= np.hypot(*(lumen - marks[k_]).T) > a.fornix_r
    for nm, msk in (("papilla", side), ("wall", ~side)):
        row[f"max_e1_{nm}"] = round(float(e1[li][msk].max()), 5)
        row[f"median_e1_{nm}"] = round(float(np.median(e1[li][msk & away])), 5)
        row[f"p95_e1_{nm}"] = round(float(np.percentile(e1[li][msk & away], 95)), 5)
    print(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+", help="base paths without extension")
    ap.add_argument("--out", default="fea")
    ap.add_argument("--tag", default="", help="suffix for case names (e.g. mesh study)")
    ap.add_argument("--pressure", type=float, default=0.0002, help="MPa (0.0002 = 200 Pa)")
    ap.add_argument("--E", type=float, default=0.014, help="Young's modulus, MPa")
    ap.add_argument("--nu", type=float, default=0.49)
    ap.add_argument("--lumen-h", dest="lumen_h0", type=float, default=0.002,
                    help="smallest lumen edge, used at curves and fornices, mm (default 0.002)")
    ap.add_argument("--lumen-hmax", dest="lumen_hmax0", type=float, default=0.02,
                    help="largest lumen edge on straight parts, mm (default 0.02); "
                         "set equal to --lumen-h for uniform sizing")
    ap.add_argument("--curv-k", type=float, default=0.3,
                    help="edge length = k x local radius of curvature (default 0.3)")
    ap.add_argument("--outer-h", dest="outer_h0", type=float, default=0.03, help="outer edge size, mm")
    ap.add_argument("--esize", dest="esize0", type=float, default=0.05, help="interior element size, mm")
    ap.add_argument("--mesh-scale", type=float, default=1.0,
                    help="scale all mesh sizes (mesh study: e.g. 2, 1, 0.5)")
    ap.add_argument("--nsub", type=int, default=20, help="initial substeps")
    ap.add_argument("--fornix-r", type=float, default=0.05, help="fornix region radius, mm")
    ap.add_argument("--fornix-mean-r", type=float, default=0.025,
                    help="radius (mm) of the area-weighted mean fornix stress (SCFmean), default 0.025")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--port", type=int, default=0,
                    help="gRPC port for MAPDL (give each simultaneous run its own port; 0 = default)")
    ap.add_argument("--start-timeout", type=int, default=300,
                    help="seconds to wait for MAPDL to start (default 300; HPC licence checkout can be slow)")
    ap.add_argument("--mapdl-switches", default="-smp",
                    help="extra MAPDL command-line switches (default '-smp': shared-memory, no MPI)")
    ap.add_argument("--exec-file", help="path to the MAPDL executable if it is not found "
                    r"automatically, e.g. C:\Program Files\ANSYS Inc\v261\ansys\bin\winx64\ANSYS261.exe")
    ap.add_argument("--dry-run", action="store_true", help="write .inp files only")
    ap.add_argument("--fields", default="seqv_over_p,e1,hydro_over_p,disp_um",
                    help="contour maps to save: " + ", ".join(FIELDS))
    ap.add_argument("--clim", default="",
                    help="fixed colour limits shared by all cases, e.g. 'seqv_over_p=0,8;e1=0,0.06'")
    ap.add_argument("--deformed", action="store_true", help="plot on the deformed shape (true scale)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--title-pressure-pa", type=float, default=0,
                    help="pressure shown in plot titles (Pa); use when a run used scaled units "
                         "(e.g. --E 14 --pressure 0.05 = kPa units -> give 50)")
    ap.add_argument("--fmt", default="png", help="figure formats, e.g. 'png,pdf' (pdf/svg for the journal)")
    ap.add_argument("--no-plots", action="store_true", help="save the field files only")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute papilla and boundary metrics from saved <case>_fields.npz (no MAPDL)")
    ap.add_argument("--replot", action="store_true",
                    help="re-draw contour maps from saved <case>_fields.npz files (no MAPDL); "
                         "give the npz paths as cases")
    a = ap.parse_args()

    if a.replot:
        for f in a.cases:
            plot_case(f if f.endswith(".npz") else f + "_fields.npz", a)
        return
    if a.rescore:
        rows = [rescore(f if f.endswith(".npz") else f + "_fields.npz", a) for f in a.cases]
        out = os.path.join(a.out, "summary_rescored.csv")
        keys = list(rows[0])
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"written {out}")
        return
    mapdl = None
    if not a.dry_run:
        from ansys.mapdl.core import launch_mapdl
        kw = {"exec_file": a.exec_file} if a.exec_file else {}
        if a.port:
            kw["port"] = a.port
        run_dir = os.path.abspath(os.path.join(a.out, "mapdl_run"))
        os.makedirs(run_dir, exist_ok=True)
        print(f"launching MAPDL (timeout {a.start_timeout} s, switches '{a.mapdl_switches}', "
              f"working directory {run_dir}) ...", flush=True)
        mapdl = launch_mapdl(nproc=a.nproc, override=True, start_timeout=a.start_timeout,
                             additional_switches=a.mapdl_switches, run_location=run_dir, **kw)
        print("MAPDL started", flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "summary.csv")
    fields = ["case", "mesh_scale", "pressure_MPa", "load_factor", "converged", "E_MPa", "nu", "lumen_h_mm",
              "n_elem", "SCF_FORNIX_L", "SCF_FORNIX_R", "SCFmean_FORNIX_L", "SCFmean_FORNIX_R", "SCF_global", "papilla_hydro_over_p",
              "papilla_hydro_median_over_p", "papilla_area_mm2",
              "tip_disp_um", "lumen_area0_mm2", "dA_over_A0_per_kPa", "max_e1_papilla",
              "max_e1_wall", "median_e1_papilla", "median_e1_wall", "p95_e1_papilla", "p95_e1_wall"]
    new = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            for base in a.cases:
                try:
                    run_case(mapdl, base, a, w)
                    f.flush()
                except Exception as exc:
                    print(f"{base}: FAILED - {exc}")
    finally:
        if mapdl is not None:
            mapdl.exit()


if __name__ == "__main__":
    main()
