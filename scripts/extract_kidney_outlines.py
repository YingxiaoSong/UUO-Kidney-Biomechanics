"""
Extract kidney outer outline and renal pelvis lumen from H&E panels in a PDF figure,
scale them to mm using the PDF's scale bar, and export DXF (splines), CSV, overlay PNG
and a JSON of basic dimensions.

Requirements:
    pip install pymupdf scikit-image scipy numpy pillow ezdxf matplotlib shapely

Usage examples:
    python extract_kidney_outlines.py H_E_Cross_Sections_-_UUO.pdf --panel UUO_POD3 --cut-col 1120
    python extract_kidney_outlines.py H_E_Cross_Sections_-_UUO.pdf --panel UUO_POD7 --lumen-mode hull
    python extract_kidney_outlines.py H_E_Cross_Sections_-_UUO.pdf --panel all
    python extract_kidney_outlines.py fig.pdf --center 314 120 --name mypanel --lumen-mode hull

Tested settings for H_E_Cross_Sections_-_UUO.pdf:
    UUO_POD3: --cut-col 1120            (splits it from the neighboring section)
    UUO_POD7: --lumen-mode hull         (cavity opens at the hilum; hull seals the mouth)
    Others: outer outline is fine, but the pelvic space is a thin slit that is often
    open to the hilum, so the automatic lumen pick is unreliable. Check the overlay;
    try --lumen-seed X Y (pt) and/or --lumen-mode hull, or trace the lumen by hand.

Openings and thin slits:
    --seal X1 Y1 X2 Y2   draw a seal line (pt) across an opening such as the hilum mouth,
                         so the pelvic space becomes a closed cavity (repeat for several).
    --pick-seal          click the seal lines on the image instead (pairs of points);
                         they are saved to <name>_seal_pts.txt and printed for reuse.
    --cut X1 Y1 X2 Y2    cut line (pt) through the gap between this section and a touching
                         neighbour, in any direction; --pick-cut to click it. Replaces
                         --cut-col/--cut-row, which only cut along one pixel column/row.
    --close-r R          morphological closing (px) to seal tiny breaks in the tissue.
    --crack-seed X Y     a point (pt) inside a slit too thin to model as a cavity; its
                         centerline is exported on layer CRACK as an open spline. In Fusion,
                         split the body along it; in ANSYS, pressurize both crack faces and
                         add frictionless contact. Keep the crack inside the tissue: seal the
                         hilum mouth so the crack does not reach the outer surface.

Slot model for collapsed or partly visible pelvic spaces (recommended for POD1, POD2,
sham and contralateral sections):
    --slot-pts X1 Y1 ... Xn Yn   centerline points (pt) from one fornix, around the papilla
                                 tip, to the other fornix; or use --pick-slot to click them.
    --slot-width W               slot width in um (default 8). The centerline is smoothed by
                                 an interpolating spline and offset by W/2 on each side; the
                                 ends are closed by semicircular caps (layer PELVIS_LUMEN:
                                 two splines + two arcs). Clicked points are saved to
                                 <name>_slot_pts.txt so the geometry can be regenerated exactly.

Hybrid sections (partly open, partly collapsed):
    --hybrid             merge the slot with the open cavities marked by --lumen-seed
                         (one X Y pair inside each open part; unmarked cavities such as
                         the hilar branch are left out).
                         Click the full centerline from fornix to fornix as for a slot
                         (--pick-slot), along the middle of the open parts; where the space
                         is open, the real cavity takes over. Seal an opening first (one seal
                         line per opening; both ends must touch tissue).
                         Junctions are rounded (--junction-r, default half the slot width).
                         Without seeds, only the largest cavity the slot crosses is merged.

Final editing:
    --edit               open an editor on the image before export. Keys: 1 lumen / 2 kidney
                         outline; e erase, a add (click a polygon), c connect (click a line that
                         becomes a slot, e.g. to extend the pelvic space to the next open part);
                         Enter apply, Esc cancel, z undo, q finish. Edits are saved to
                         <name>_edits.json; --edits FILE re-applies them without the editor.

Smoothing for meshing:
    --round-um R         round every outline (kidney and open cavities, not the slot):
                         features narrower than 2R are removed and all corners get a
                         radius of at least R, so the mesh needs fewer tiny elements.
                         The fornix is rounded too, so R doubles as the fornix fillet in
                         the sensitivity study (e.g. 0, 5, 10 um). Typical: 5 um.
    --outer-round-um R   separate (usually larger) radius for the outer outline, e.g. 40, to
                         remove tears, folds and small notches from sectioning. The outer
                         surface is far from the pelvis, so this barely affects the results.
    --outline-sigma / --lumen-sigma   pixel smoothing before contouring (defaults 4 / 2.5).

Thin slits below the image resolution cannot be segmented reliably. Each run also
saves <name>_canvas.png and a CANVAS_FRAME rectangle in the DXF: in Fusion, insert the
PNG as a canvas, snap its corners to the frame, and trace the slit by hand as a spline.
--smooth 3 --thresh 620 helps pale, partly open clefts (e.g. UUO_POD1) read as cavity.

Coordinates: output in mm, origin at the papilla tip (falls back to the kidney centroid
if the tip cannot be found), x to the right, y up (image orientation as in the PDF).
"""
import argparse, csv, json, os, warnings
import numpy as np
import pymupdf
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology, draw
import ezdxf

warnings.filterwarnings("ignore")

# Approximate panel centers in PDF points (x, y) for H_E_Cross_Sections_-_UUO.pdf
PANELS = {
    "UUO_POD1": (74, 111), "UUO_POD2": (194, 117), "UUO_POD3": (314, 120),
    "UUO_POD7": (431, 115), "UUO_SHAM": (560, 113),
    "CL_POD1": (74, 253), "CL_POD2": (192, 251), "CL_POD3": (311, 253),
    "CL_POD7": (415, 253), "CL_SHAM": (540, 255),
}


def find_scale_bar_pt(page):
    """Length (pt) of the longest short horizontal vector line on the page."""
    best = 0.0
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 0.5 and 5 < abs(p2.x - p1.x) < 200:
                    best = max(best, abs(p2.x - p1.x))
    if best == 0:
        raise RuntimeError("Scale bar not found; pass --bar-pt")
    return best


def render(page, center, half, dpi):
    cx, cy = center
    clip = pymupdf.Rect(cx - half, cy - half, cx + half, cy + half) & page.rect
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    to_px = lambda x, y: (int((y - clip.y0) * dpi / 72), int((x - clip.x0) * dpi / 72))  # (row, col)
    from_px = lambda c, r: (clip.x0 + c * 72 / dpi, clip.y0 + r * 72 / dpi)            # -> (x, y) pt
    return img.astype(float), to_px(cx, cy), to_px, from_px


def segment(img, seed, args, lumen_seed=None, seals_px=(), cuts_px=()):
    inten = img.sum(2)
    # black figure annotations (scale bar, panel letters) are not tissue: treat them as background
    inten = np.where(inten < 120, 765.0, inten)
    if args.smooth > 0:   # average out tubule lumens so only real cavities stay bright
        inten = ndi.gaussian_filter(inten, args.smooth)
    tissue = inten < args.thresh
    tissue = morphology.remove_small_objects(tissue, max_size=args.min_obj)
    if args.close_r > 0:   # seal tiny breaks in the tissue so cavities become closed
        tissue = morphology.binary_closing(tissue, morphology.disk(args.close_r))
    if seals_px:           # user-drawn seal lines, e.g. across the hilum mouth
        line = np.zeros_like(tissue)
        for r0, c0, r1, c1 in seals_px:
            # extend each seal a little past both clicked ends so it reaches into tissue
            L = max(np.hypot(r1 - r0, c1 - c0), 1)
            er, ec = (r1 - r0) / L * args.seal_extend, (c1 - c0) / L * args.seal_extend
            r0, c0, r1, c1 = int(r0 - er), int(c0 - ec), int(r1 + er), int(c1 + ec)
            rr, cc = draw.line(r0, c0, r1, c1)
            ok = (rr >= 0) & (rr < line.shape[0]) & (cc >= 0) & (cc < line.shape[1])
            line[rr[ok], cc[ok]] = True
        tissue = tissue | morphology.binary_dilation(line, morphology.disk(2))
    filled_all = ndi.binary_fill_holes(tissue)
    for c in args.cut_col or []:            # optional manual cuts (pixels in rendered crop)
        filled_all[:, c:] = False if c > seed[1] else filled_all[:, c:]
        if c <= seed[1]:
            filled_all[:, :c] = False
    for r in args.cut_row or []:
        if r > seed[0]:
            filled_all[r:, :] = False
        else:
            filled_all[:r, :] = False
    if cuts_px:   # cut lines between this section and a touching neighbour
        line = np.zeros_like(filled_all)
        for r0, c0, r1, c1 in cuts_px:
            L = max(np.hypot(r1 - r0, c1 - c0), 1)       # extend past the clicked ends
            er, ec = (r1 - r0) / L * 15, (c1 - c0) / L * 15
            rr, cc = draw.line(int(r0 - er), int(c0 - ec), int(r1 + er), int(c1 + ec))
            ok = (rr >= 0) & (rr < line.shape[0]) & (cc >= 0) & (cc < line.shape[1])
            line[rr[ok], cc[ok]] = True
        filled_all &= ~morphology.binary_dilation(line, morphology.disk(3))
    opened = morphology.binary_opening(filled_all, morphology.disk(args.open_r))
    lab, _ = ndi.label(opened)
    s = lab[seed]
    if s == 0:  # seed fell in a hole/background: take the nearest labeled component
        idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        s = lab[idx[0][seed], idx[1][seed]]
    kid = morphology.binary_dilation(lab == s, morphology.disk(args.open_r)) & filled_all
    kid = ndi.binary_fill_holes(kid)
    l2, _ = ndi.label(kid)
    kid = l2 == np.bincount(l2.ravel())[1:].argmax() + 1

    if args.lumen_mode == "hole":
        cand = kid & ~tissue
    else:  # hull: also captures a cavity that opens to the outside (e.g. POD7 hilum)
        cand = morphology.convex_hull_image(kid) & ~tissue
        cand = morphology.binary_opening(cand, morphology.disk(3))
    hl, n = ndi.label(cand)
    if n == 0:
        return kid, tissue, None, None
    areas = np.bincount(hl.ravel())[1:]
    order = np.argsort(-areas)
    if lumen_seed is not None and hl[lumen_seed] > 0:   # user-picked cavity
        lid = hl[lumen_seed]
        lumen = hl == lid
        rest = [i + 1 for i in order if i + 1 != lid]
        second = hl == rest[0] if rest else None
    else:
        if lumen_seed is not None:
            print("  warning: --lumen-seed is not inside a cavity; using the largest one")
        lumen = hl == order[0] + 1
        second = hl == order[1] + 1 if n > 1 else None
    if args.lumen_mode == "hull":
        kid = kid | lumen  # closed kidney outline includes the sealed cavity mouth
    return kid, tissue, lumen, second


def landmarks(lumen, tissue):
    try:
        hull = morphology.convex_hull_image(lumen)
        pap = hull & ~lumen & tissue
        pl, n = ndi.label(pap)
        if n == 0:
            return None, None, None
        pap = pl == np.bincount(pl.ravel())[1:].argmax() + 1
        py, px = np.nonzero(pap)
        ty = py.min()
        tip = (float(px[py == ty].mean()), float(ty))
        tx = int(tip[0])
        ly, lx = np.nonzero(lumen[:, :tx])
        ry, rx = np.nonzero(lumen[:, tx:])
        fL = (float(lx[ly.argmax()]), float(ly.max())) if len(ly) else None
        fR = (float(rx[ry.argmax()] + tx), float(ry.max())) if len(ry) else None
        return tip, fL, fR
    except Exception:
        return None, None, None


def contour_pts(mask, sigma, tol):
    sm = ndi.gaussian_filter(mask.astype(float), sigma)
    c = max(measure.find_contours(sm, 0.5), key=len)
    c = measure.approximate_polygon(c, tol)[:-1]
    return [(p[1], p[0]) for p in c]  # (col, row)


def pick_points(img, title):
    """Click points along the pelvic space; Enter or right-click to finish."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img.astype(np.uint8))
    ax.set_title(title + "\nLeft-click along the pelvic space from one fornix, around the papilla "
                 "tip, to the other fornix.\nRight-click removes the last point; Enter finishes.",
                 fontsize=9)
    pts = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=None)
    plt.close(fig)
    return pts  # list of (col, row)


def make_slot(center_mm, width_mm, n=120):
    """Smooth the centerline and build a slot with rounded ends.
    Returns (centerline, left offset, right offset, start cap, end cap); caps are
    (center, radius, start_deg, end_deg) for DXF arcs drawn counterclockwise."""
    from scipy.interpolate import splprep, splev
    p = np.asarray(center_mm, float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]
    p = p[keep]
    k = min(3, len(p) - 1)
    tck, _ = splprep([p[:, 0], p[:, 1]], s=0, k=k)
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    t = np.c_[dx, dy] / np.hypot(dx, dy)[:, None]
    nrm = np.c_[-t[:, 1], t[:, 0]]
    c = np.c_[x, y]
    r = width_mm / 2
    left, right = c + r * nrm, c - r * nrm
    a0 = np.degrees(np.arctan2(nrm[0, 1], nrm[0, 0]))
    a1 = np.degrees(np.arctan2(nrm[-1, 1], nrm[-1, 0]))
    cap0 = (tuple(c[0]), r, a0, a0 + 180)          # from left offset, around the back, to right
    cap1 = (tuple(c[-1]), r, a1 + 180, a1 + 360)    # from right offset, around the front, to left
    return c, left, right, cap0, cap1


def round_outline(pts_mm, r_mm, spacing=None):
    """Remove features narrower than 2r and round every corner to a radius of at least r:
    morphological opening (cuts thin spikes, rounds convex corners) followed by closing
    (fills thin notches, rounds concave corners). Returns resampled outline points."""
    from shapely.geometry import Polygon
    from scipy.interpolate import splprep, splev
    v = np.asarray(pts_mm, float)
    v = v[np.r_[True, np.hypot(*np.diff(v, axis=0).T) > 1e-9]]
    try:   # follow the same smooth periodic spline the DXF would show, not the raw polygon
        tck, _ = splprep([v[:, 0], v[:, 1]], s=0, per=1, k=3)
        L = float(np.sum(np.hypot(*np.diff(np.vstack([v, v[:1]]), axis=0).T)))
        x, y = splev(np.linspace(0, 1, max(int(L / 0.002), 200), endpoint=False), tck)
        v = np.c_[x, y]
    except Exception:
        pass
    p = Polygon(v).buffer(0)
    q = p.buffer(-r_mm, quad_segs=16).buffer(r_mm, quad_segs=16)
    q = q.buffer(r_mm, quad_segs=16).buffer(-r_mm, quad_segs=16)
    if q.is_empty:
        print(f"  warning: rounding radius {r_mm * 1000:.1f} um removed a whole outline; kept it unrounded")
        return pts_mm
    if q.geom_type == "MultiPolygon":
        q = max(q.geoms, key=lambda g: g.area)
    ext = np.array(q.exterior.coords)[:-1]
    h = spacing or max(min(r_mm / 2, 0.01), 0.001)
    closed = np.vstack([ext, ext[:1]])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    si = np.linspace(0, s[-1], max(int(s[-1] / h), 24), endpoint=False)
    return [tuple(p) for p in np.c_[np.interp(si, s, closed[:, 0]), np.interp(si, s, closed[:, 1])]]


def enforce_wall(cav_pts, outer_pts, wall_mm, seals_mm, r_mm=0.005):
    """Where a seal line closes an opening, keep at least wall_mm of tissue between the
    cavity and the outer surface, so the seal becomes a thin wall instead of a zero-thickness
    contact. Only the region around the seal lines is changed. Corners are rounded (r_mm)."""
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
    if not seals_mm:
        return cav_pts, False
    cav, out = Polygon(cav_pts).buffer(0), Polygon(outer_pts).buffer(0)
    zone = unary_union([LineString(sl).buffer(4 * wall_mm + 0.05) for sl in seals_mm])
    band = out.difference(out.buffer(-wall_mm, quad_segs=16)).intersection(zone)
    band = band.union(zone.difference(out))          # anything beyond the outer surface too
    if not cav.intersects(band):
        return cav_pts, False
    c = cav.difference(band)
    if c.is_empty:
        return cav_pts, False
    # round the clipped corners only inside the seal zone, so thin slots elsewhere are kept
    near = c.intersection(zone).buffer(-r_mm, quad_segs=16).buffer(r_mm, quad_segs=16)
    c = c.difference(zone).union(near).buffer(1e-7).buffer(-1e-7)
    if c.geom_type != "Polygon":
        parts = sorted(getattr(c, "geoms", [c]), key=lambda g: g.area, reverse=True)
        lost = sum(g.area for g in parts[1:])
        print(f"  WARNING: keeping a {wall_mm * 1000:g} um wall at the seal split the pelvic space into "
              f"{len(parts)} pieces; only the largest was kept ({lost * 1e6:.0f} um^2 dropped). Move the "
              "slot / seal further from the outer surface, or lower --min-wall-um.")
        c = parts[0]
    ext = np.array(c.exterior.coords)[:-1]
    closed = np.vstack([ext, ext[:1]])
    s_ = np.r_[0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    si = np.linspace(0, s_[-1], max(int(s_[-1] / 0.002), 24), endpoint=False)
    pts = [tuple(p) for p in np.c_[np.interp(si, s_, closed[:, 0]), np.interp(si, s_, closed[:, 1])]]
    return pts, True


def poly_area_mm(pts):
    p = np.asarray(pts)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def merge_hybrid(cavities_mm, center_mm, width_mm, fillet_mm, n_seeded=1, round_mm=0):
    """Union of the slot around the centerline with every automatically segmented open
    cavity it passes through; concave junctions are rounded by a morphological closing of
    radius fillet_mm. Returns the merged outline and the number of cavities merged."""
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
    slot = LineString(center_mm).buffer(width_mm / 2, quad_segs=16)
    if round_mm > 0:   # round the open cavities first; the slot itself is left as defined
        cavities_mm = [round_outline(c, round_mm) for c in cavities_mm if len(c) >= 3]
    cavs = [Polygon(c).buffer(0) for c in cavities_mm if len(c) >= 3]
    cavs = [c for c in cavs if c.intersects(slot)]
    if cavs and n_seeded == 0:
        cavs = [max(cavs, key=lambda c: c.area)]
    if not cavs:
        raise SystemExit("The slot does not pass through any open cavity: run the centerline along "
                         "the middle of the open part of the pelvic space, check the lumen seed, and "
                         "make sure the seal line closes the opening (each end must touch tissue).")
    u = unary_union(cavs + [slot])
    if fillet_mm > 0:
        u = u.buffer(fillet_mm, quad_segs=16).buffer(-fillet_mm, quad_segs=16)
    if u.geom_type == "MultiPolygon":
        print(f"  WARNING: the slot and the open parts form {len(u.geoms)} separate pieces; only the "
              "piece containing most of the slot is kept. Make the centerline run through each open part.")
        u = max(u.geoms, key=lambda g: g.intersection(slot).area)
    ext = np.array(u.exterior.coords)[:-1]
    h = min(0.002, width_mm / 4)
    closed = np.vstack([ext, ext[:1]])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    si = np.linspace(0, s[-1], max(int(s[-1] / h), 24), endpoint=False)
    pts = np.c_[np.interp(si, s, closed[:, 0]), np.interp(si, s, closed[:, 1])]
    return {"pts": pts, "area": u.area, "n_cavities": len(cavs)}


def spline_is_simple(sp, width_mm):
    from shapely.geometry import LinearRing
    try:
        q = [(p.x, p.y) for p in sp.flattening(width_mm / 20)]
        return LinearRing(q).is_simple
    except Exception:
        return False


# ------------------------------------------------------------------ final editing ----
def _largest(g):
    if g.is_empty:
        return g
    if g.geom_type != "Polygon":
        g = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
    return g


def _resample_poly(g, h=0.002):
    ext = np.array(g.exterior.coords)[:-1]
    closed = np.vstack([ext, ext[:1]])
    s_ = np.r_[0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    si = np.linspace(0, s_[-1], max(int(s_[-1] / h), 24), endpoint=False)
    return [tuple(p) for p in np.c_[np.interp(si, s_, closed[:, 0]), np.interp(si, s_, closed[:, 1])]]


def apply_ops(state, ops, pt2mm, width_mm, round_mm, outer_round_mm=None):
    """Apply recorded edits to the outlines (shapely polygons in mm).
    ops: [{"op": "erase"|"add"|"connect", "target": "lumen"|"outer", "pts_pt": [[x, y], ...]}]
      erase   - remove the drawn polygon from the target
      add     - add the drawn polygon to the target
      connect - add a slot (current slot width) along the drawn polyline, e.g. to extend the
                pelvic space to the next open part or to a fornix
    Corners created by erase/add are rounded locally (radius round_mm)."""
    from shapely.geometry import Polygon, LineString
    st = dict(state)
    for op in ops:
        pts = [tuple(p) for p in op.get("pts_mm", [])] or [pt2mm(*p) for p in op["pts_pt"]]
        tgt = op.get("target", "lumen")
        g = st.get(tgt)
        if op["op"] == "connect":
            if len(pts) < 2:
                continue
            piece = LineString(pts).buffer(width_mm / 2, quad_segs=16)
            g = piece if g is None or g.is_empty else g.union(piece)
            r = width_mm / 2
            g = g.buffer(r, quad_segs=16).buffer(-r, quad_segs=16)       # round the junctions
        else:
            if len(pts) < 3 or g is None:
                continue
            poly = Polygon(pts).buffer(0)
            g = g.difference(poly) if op["op"] == "erase" else g.union(poly)
            if tgt == "outer":   # the outer surface is far from the pelvis: round it globally
                rr_ = outer_round_mm or round_mm
                if rr_ > 0:
                    g = _largest(g.buffer(1e-7))
                    g = g.buffer(-rr_, quad_segs=16).buffer(rr_, quad_segs=16)
                    g = g.buffer(rr_, quad_segs=16).buffer(-rr_, quad_segs=16)
            elif round_mm > 0:   # lumen: round the new corners only near the drawn polygon
                rr_ = round_mm
                zone = poly.boundary.buffer(3 * rr_)
                big = zone.buffer(3 * rr_)
                near = g.intersection(big)
                near = near.buffer(-rr_, quad_segs=16).buffer(rr_, quad_segs=16)
                near = near.buffer(rr_, quad_segs=16).buffer(-rr_, quad_segs=16)
                g = g.difference(zone).union(near.intersection(zone))
        st[tgt] = _largest(g.buffer(1e-7).buffer(-1e-7))
    return st


def interactive_editor(img, state, ops, mm2px, px2pt, pt2mm, width_mm, round_mm, title,
                       outer_round_mm=None):
    """Edit the final outlines on the image. Keys:
       1 / 2  edit the pelvic lumen / the kidney outline
       e      erase: click a polygon around the part to delete, Enter
       a      add: click a polygon to add, Enter
       c      connect: click a line (e.g. from the end of the cavity to the next open part
              or to a fornix); it becomes a slot of the current width, Enter
       Esc cancel the current drawing, z undo the last edit, q (or close) finish.
       Left-click adds a point, right-click removes the last point. The zoom/pan tools work;
       turn them off again before clicking points."""
    import matplotlib.pyplot as plt
    hist = []
    S = {"target": "lumen", "mode": None, "pts": [], "art": []}
    keys = {"keymap.save": [], "keymap.back": ["left", "backspace"], "keymap.quit": ["q"],
            "keymap.forward": ["right"], "keymap.home": ["h", "home"]}
    with plt.rc_context(keys):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img.astype(np.uint8))

        def outline(g):
            if g is None or g.is_empty:
                return np.zeros((0, 2))
            return np.array([mm2px(x, y) for x, y in np.array(g.exterior.coords)])

        def redraw():
            for a in S["art"]:
                a.remove()
            S["art"] = []
            o, l = outline(S_state["outer"]), outline(S_state["lumen"])
            if len(o):
                S["art"] += ax.plot(o[:, 0], o[:, 1], "-", color="k", lw=1)
            if len(l):
                S["art"] += ax.plot(l[:, 0], l[:, 1], "-", color="m", lw=1)
            if S["pts"]:
                p = np.array(S["pts"] + ([S["pts"][0]] if S["mode"] in ("erase", "add") else []))
                S["art"] += ax.plot(p[:, 0], p[:, 1], "o-", color="orange", ms=3, lw=1)
            mode = S["mode"] or "none"
            ax.set_title(f"{title} | target: {S['target']} | mode: {mode} | edits: {len(ops)}\n"
                         "1 lumen, 2 outline | e erase, a add, c connect | Enter apply, Esc cancel, "
                         "z undo, q finish", fontsize=9)
            fig.canvas.draw_idle()

        def on_click(ev):
            tb = getattr(fig.canvas.manager, "toolbar", None)
            if ev.inaxes != ax or S["mode"] is None or (tb is not None and getattr(tb, "mode", "")):
                return
            if ev.button == 1:
                S["pts"].append((ev.xdata, ev.ydata))
            elif ev.button == 3 and S["pts"]:
                S["pts"].pop()
            redraw()

        def on_key(ev):
            k = ev.key
            if k in ("1", "2"):
                S["target"] = "lumen" if k == "1" else "outer"
            elif k in ("e", "a", "c"):
                S["mode"] = {"e": "erase", "a": "add", "c": "connect"}[k]
                S["pts"] = []
            elif k == "escape":
                S["pts"] = []
                S["mode"] = None
            elif k == "enter" and S["mode"] and S["pts"]:
                op = {"op": S["mode"], "target": S["target"],
                      "pts_pt": [list(px2pt(x, y)) for x, y in S["pts"]]}
                hist.append(dict(S_state))
                new = apply_ops(S_state, [op], pt2mm, width_mm, round_mm, outer_round_mm)
                S_state.update(new)
                ops.append(op)
                S["pts"] = []
            elif k == "z" and hist:
                S_state.update(hist.pop())
                ops.pop()
            redraw()

        S_state = dict(state)
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        plt.show()
    return S_state, ops


def skeleton_path(mask):
    """Longest path through the skeleton of a thin region, as an ordered (col, row) list."""
    sk = morphology.skeletonize(mask)
    pts = set(zip(*np.nonzero(sk)))
    if len(pts) < 2:
        return None
    nbr = lambda p: [(p[0] + i, p[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i or j) and (p[0] + i, p[1] + j) in pts]
    def bfs(start):
        par, order = {start: None}, [start]
        for p in order:
            for q in nbr(p):
                if q not in par:
                    par[q] = p
                    order.append(q)
        return order[-1], par
    a, _ = bfs(next(iter(pts)))
    b, par = bfs(a)
    path, p = [], b
    while p is not None:
        path.append(p)
        p = par[p]
    path = measure.approximate_polygon(np.array(path, float), 1.5)
    return [(p[1], p[0]) for p in path]


def process(page, name, center, args, um_px):
    img, seed, to_px, from_px = render(page, center, args.half, args.dpi)
    if args.pick_seed:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(img.astype(np.uint8))
        ax.set_title(name + "\nClick once inside each white open part of the pelvic space to include "
                     "(one point per part).\nRight-click removes the last point; Enter finishes.", fontsize=9)
        clicked = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=None)
        plt.close(fig)
        if clicked:
            pts = [from_px(c, r) for c, r in clicked]
            args.lumen_seed = [v for p in pts for v in p]
            txt = " ".join(f"{v:.2f}" for v in args.lumen_seed)
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, name + "_seed_pts.txt"), "w") as f:
                f.write(txt + "\n")
            print("  lumen seeds (pt), reuse with --lumen-seed: " + txt)
    if args.lumen_seed and len(args.lumen_seed) % 2:
        raise SystemExit("--lumen-seed needs X Y pairs.")
    seeds_pt = list(zip(args.lumen_seed[0::2], args.lumen_seed[1::2])) if args.lumen_seed else []
    ls = to_px(*seeds_pt[0]) if seeds_pt else None
    if args.pick_cut:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(img.astype(np.uint8))
        ax.set_title(name + "\nClick two points to cut this section off a touching neighbour "
                     "(pairs of points; draw through the gap).\nRight-click removes the last "
                     "point; Enter finishes.", fontsize=9)
        clicked = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=None)
        plt.close(fig)
        if len(clicked) < 2 or len(clicked) % 2:
            raise SystemExit("Cut lines need pairs of points (2, 4, ...).")
        pts = [from_px(c, r) for c, r in clicked]
        new = [[*pts[i], *pts[i + 1]] for i in range(0, len(pts), 2)]
        args.cut = (args.cut or []) + new
        txt = "; ".join(" ".join(f"{v:.2f}" for v in s4) for s4 in new)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, name + "_cut_pts.txt"), "w") as f:
            f.write(txt + "\n")
        print("  cut lines (pt), reuse with --cut: " + txt)
    cuts = [to_px(x1, y1) + to_px(x2, y2) for x1, y1, x2, y2 in (args.cut or [])]
    if args.pick_seal:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(img.astype(np.uint8))
        ax.set_title(name + "\nClick the two ends of each opening to seal (e.g. across the hilum mouth),"
                     " in pairs.\nRight-click removes the last point; Enter finishes.", fontsize=9)
        clicked = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=None)
        plt.close(fig)
        if len(clicked) < 2 or len(clicked) % 2:
            raise SystemExit("Seal lines need pairs of points (2, 4, 6, ...).")
        pts = [from_px(c, r) for c, r in clicked]
        new = [[*pts[i], *pts[i + 1]] for i in range(0, len(pts), 2)]
        args.seal = (args.seal or []) + new
        txt = "; ".join(" ".join(f"{v:.2f}" for v in s4) for s4 in new)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, name + "_seal_pts.txt"), "w") as f:
            f.write(txt + "\n")
        print("  seal lines (pt), reuse with --seal: " + txt)
    seals = [to_px(x1, y1) + to_px(x2, y2) for x1, y1, x2, y2 in (args.seal or [])]
    kid, tissue, lumen, second = segment(img, seed, args, ls, seals, cuts)
    crack, crack_mask = None, None
    if args.crack_seed:   # export this thin cavity as a centerline (crack) instead of an outline
        cl, _ = ndi.label(kid & ~tissue)
        r, c = to_px(*args.crack_seed)
        win = cl[max(r - 6, 0):r + 7, max(c - 6, 0):c + 7]
        ids = np.bincount(win[win > 0].ravel()) if (win > 0).any() else None
        if ids is None:
            print("  warning: --crack-seed is not inside a cavity; no crack exported")
        else:
            crack_mask = cl == ids.argmax()
            crack = skeleton_path(crack_mask)
            if lumen is not None and (lumen & crack_mask).any():
                lumen = None   # the crack replaces this cavity
            if second is not None and (second & crack_mask).any():
                second = None
    tip, fL, fR = landmarks(lumen, tissue) if lumen is not None else (None, None, None)
    slot_px = None
    if args.pick_slot:
        clicked = pick_points(img, name)
        if len(clicked) < 3:
            raise SystemExit("Need at least 3 points for a slot.")
        pts_pt = [from_px(c, r) for c, r in clicked]
        flat = " ".join(f"{x:.2f} {y:.2f}" for x, y in pts_pt)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, name + "_slot_pts.txt"), "w") as f:
            f.write(flat + "\n")
        print("  slot points (pt), reuse with --slot-pts: " + flat)
        args.slot_pts = [v for xy in pts_pt for v in xy]
    if args.slot_pts:
        v = args.slot_pts
        if len(v) % 2 or len(v) < 6:
            raise SystemExit("--slot-pts needs at least 3 X Y pairs.")
        slot_px = []
        for x, y in zip(v[0::2], v[1::2]):
            r, c = to_px(x, y)
            slot_px.append((c, r))
        if args.hybrid:
            second = None            # keep the automatic cavity; it is merged with the slot
        else:
            lumen, second = None, None   # the slot replaces any automatic cavity
        a, b = np.array(slot_px[0], float), np.array(slot_px[-1], float)
        mid = (a + b) / 2
        far = max(slot_px, key=lambda p: np.hypot(p[0] - mid[0], p[1] - mid[1]))
        tip, fL, fR = (float(far[0]), float(far[1])), tuple(a), tuple(b)
    if tip is None:
        ys, xs = np.nonzero(kid)
        origin = (xs.mean(), ys.mean())
    else:
        origin = tip
    mm = um_px / 1000.0
    to_mm = lambda c, r: ((c - origin[0]) * mm, -(r - origin[1]) * mm)

    shapes = {"KIDNEY_OUTER": contour_pts(kid, args.outline_sigma, 3)}
    if lumen is not None:
        shapes["PELVIS_LUMEN"] = contour_pts(lumen, args.lumen_sigma, 2)
    if second is not None and args.keep_second:
        shapes["SECOND_CAVITY"] = contour_pts(second, args.lumen_sigma, 2)
    shapes_mm = {k: [to_mm(*p) for p in v] for k, v in shapes.items()}
    round_mm = args.round_um / 1000.0
    outer_mm = (args.outer_round_um if args.outer_round_um is not None else args.round_um) / 1000.0
    if outer_mm > 0:   # outer surface: may use a larger radius to remove tears and notches
        shapes_mm["KIDNEY_OUTER"] = round_outline(shapes_mm["KIDNEY_OUTER"], outer_mm,
                                                  spacing=max(min(outer_mm / 2, 0.01), 0.002))
    if round_mm > 0 and not args.hybrid:   # hybrid rounds its cavities inside merge_hybrid
        for k in list(shapes_mm):
            if k != "KIDNEY_OUTER":
                shapes_mm[k] = round_outline(shapes_mm[k], round_mm)
    wall_mm = args.min_wall_um / 1000.0
    seals_mm = [[to_mm(c0, r0), to_mm(c1, r1)] for r0, c0, r1, c1 in seals]
    if wall_mm > 0:
        for k in ("PELVIS_LUMEN", "SECOND_CAVITY"):
            if k in shapes_mm and not args.hybrid:
                shapes_mm[k], clipped = enforce_wall(shapes_mm[k], shapes_mm["KIDNEY_OUTER"], wall_mm,
                                                     seals_mm)
                if clipped:
                    print(f"  {k}: kept a {args.min_wall_um:g} um wall to the outer surface")
    crack_mm = [to_mm(*p) for p in crack] if crack else None
    slot = None
    if slot_px:
        slot = make_slot([to_mm(*p) for p in slot_px], args.slot_width / 1000.0)
    hybrid = None
    if args.hybrid:
        if not slot:
            raise SystemExit("--hybrid needs slot points (--slot-pts or --pick-slot).")
        # open parts to merge: the cavity under each lumen seed; without seeds, the largest
        # cavity the slot passes through. Unseeded cavities (e.g. the hilar branch) stay out.
        shapes_mm.pop("PELVIS_LUMEN", None)
        hl, n = ndi.label(kid & ~tissue)
        ids = []
        for x, y in seeds_pt:
            r, c = to_px(x, y)
            win = hl[max(r - 6, 0):r + 7, max(c - 6, 0):c + 7]
            if not (win > 0).any():
                raise SystemExit(f"Lumen seed ({x}, {y}) is not inside an open cavity: check the "
                                 "seed, --thresh/--smooth, and that the seal closes the opening.")
            ids.append(int(np.bincount(win[win > 0].ravel()).argmax()))
        if ids:
            cav_mm = [[to_mm(*p) for p in contour_pts(hl == i, 1.5, 1)] for i in sorted(set(ids))]
        else:
            sizes = np.bincount(hl.ravel())
            min_px = (args.min_cavity_um2 / um_px ** 2) if args.min_cavity_um2 else 200
            cav_mm = [[to_mm(*p) for p in contour_pts(hl == i, 1.5, 1)]
                      for i in range(1, n + 1) if sizes[i] >= min_px]
            print("  hybrid: no lumen seed given; using the largest cavity the slot passes through")
        # keep the minimum wall at the seal on each sealed open cavity BEFORE joining it to the
        # slot, so the clipping can never cut the slot and disconnect the pelvic space
        if wall_mm > 0 and seals_mm:
            for i_, c_ in enumerate(cav_mm):
                c2_, clipped = enforce_wall(c_, shapes_mm["KIDNEY_OUTER"], wall_mm, seals_mm)
                if clipped:
                    cav_mm[i_] = c2_
                    print(f"  hybrid: kept a {args.min_wall_um:g} um wall between an open part and the "
                          "outer surface at the seal")
        hybrid = merge_hybrid(cav_mm, slot[0], args.slot_width / 1000.0,
                              (args.junction_r if args.junction_r else args.slot_width / 2) / 1000.0,
                              n_seeded=len(ids), round_mm=round_mm)
        if ids and hybrid["n_cavities"] < len(set(ids)):
            print("  warning: some seeded cavities do not touch the slot and were left out")
        print(f"  hybrid: merged the slot with {hybrid['n_cavities']} open cavit"
              f"{'y' if hybrid['n_cavities'] == 1 else 'ies'}")

    edited = False
    if args.edit or args.edits:
        from shapely.geometry import Polygon, LineString
        outer_poly = Polygon(shapes_mm["KIDNEY_OUTER"]).buffer(0)
        if hybrid is not None:
            lumen_poly = Polygon(hybrid["pts"]).buffer(0)
        elif slot:
            lumen_poly = LineString(slot[0]).buffer(args.slot_width / 2000.0, quad_segs=16)
        elif "PELVIS_LUMEN" in shapes_mm:
            lumen_poly = Polygon(shapes_mm["PELVIS_LUMEN"]).buffer(0)
        else:
            lumen_poly = None
        state = {"outer": outer_poly, "lumen": lumen_poly}
        locals_marks = {"fL": fL, "fR": fR}
        pt2mm = lambda x, y: to_mm(to_px(x, y)[1], to_px(x, y)[0])
        mm2px = lambda x, y: (x / mm + origin[0], origin[1] - y / mm)
        px2pt = lambda c, r: from_px(c, r)
        ops = []
        if args.edits:
            with open(args.edits) as f:
                ops = json.load(f)
            state = apply_ops(state, ops, pt2mm, args.slot_width / 1000.0, max(round_mm, 0.005),
                              max(outer_mm, 0.02))
            print(f"  applied {len(ops)} edit(s) from {args.edits}")
        if args.edit:
            state, ops = interactive_editor(img, state, ops, mm2px, px2pt, pt2mm,
                                            args.slot_width / 1000.0, max(round_mm, 0.005), name,
                                            max(outer_mm, 0.02))
        if ops:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, name + "_edits.json"), "w") as f:
                json.dump(ops, f, indent=1)
            print(f"  {len(ops)} edit(s) saved to {name}_edits.json (reuse with --edits)")
            edited = True
            if any(o.get("target") == "outer" for o in ops):
                shapes_mm["KIDNEY_OUTER"] = _resample_poly(state["outer"], 0.005)
            if state["lumen"] is not None and any(o.get("target", "lumen") == "lumen" for o in ops):
                shapes_mm.pop("PELVIS_LUMEN", None)
                hybrid = {"pts": np.array(_resample_poly(state["lumen"], min(0.002, args.slot_width / 4000.0))),
                          "area": state["lumen"].area, "n_cavities": 0}
            # a connect line that starts at a fornix extends the pelvic space: the fornix moves
            # to the far end of the line
            for o in ops:
                if o["op"] != "connect" or o.get("target", "lumen") != "lumen":
                    continue
                ends_px = [mm2px(*p) for p in o["pts_mm"]] if o.get("pts_mm") else \
                          [to_px(*p)[::-1] for p in o["pts_pt"]]
                a_, b_ = np.array(ends_px[0], float), np.array(ends_px[-1], float)
                for key in ("fL", "fR"):
                    f_ = locals_marks[key]
                    if f_ is None:
                        continue
                    if np.hypot(*(np.array(f_) - a_)) * um_px < 20:
                        locals_marks[key] = tuple(b_)
                    elif np.hypot(*(np.array(f_) - b_)) * um_px < 20:
                        locals_marks[key] = tuple(a_)
            fL, fR = locals_marks["fL"], locals_marks["fR"]
            if state["lumen"] is not None and not state["lumen"].within(state["outer"]):
                print("  warning: after editing, the lumen is not fully inside the kidney outline")

    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, name)

    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4  # mm
    msp = doc.modelspace()
    for k, pts in shapes_mm.items():
        doc.layers.add(k)
        sp = msp.add_spline(dxfattribs={"layer": k})
        sp.fit_points = pts + [pts[0]]
    if hybrid is not None:   # automatic open cavity merged with the slot: one closed curve
        pts = [tuple(p) for p in hybrid["pts"]]
        doc.layers.add("PELVIS_LUMEN")
        sp = msp.add_spline(dxfattribs={"layer": "PELVIS_LUMEN"})
        sp.fit_points = pts + [pts[0]]
        if not spline_is_simple(sp, args.slot_width / 1000.0):
            msp.delete_entity(sp)
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "PELVIS_LUMEN"})
            print("  note: spline self-intersected; hybrid lumen exported as a closed polyline")
        if slot:
            doc.layers.add("CENTERLINE")
            msp.add_spline(dxfattribs={"layer": "CENTERLINE"}).fit_points = [tuple(p) for p in slot[0]]
    elif slot:   # closed slot: two offset splines joined by two semicircular end caps
        c, left, right, cap0, cap1 = slot
        doc.layers.add("PELVIS_LUMEN")
        msp.add_spline(dxfattribs={"layer": "PELVIS_LUMEN"}).fit_points = [tuple(p) for p in left]
        msp.add_spline(dxfattribs={"layer": "PELVIS_LUMEN"}).fit_points = [tuple(p) for p in right]
        for ctr, rad, s0, s1 in (cap0, cap1):
            msp.add_arc(ctr, rad, s0, s1, dxfattribs={"layer": "PELVIS_LUMEN"})
        doc.layers.add("CENTERLINE")
        msp.add_spline(dxfattribs={"layer": "CENTERLINE"}).fit_points = [tuple(p) for p in c]
    if crack_mm:   # open curve: split the body along it in Fusion, then pressurize both faces
        doc.layers.add("CRACK")
        sp = msp.add_spline(dxfattribs={"layer": "CRACK"})
        sp.fit_points = crack_mm
    ky0, kx0 = np.nonzero(kid)
    m0 = 20
    r0, r1 = max(ky0.min() - m0, 0), min(ky0.max() + m0, img.shape[0])
    c0, c1 = max(kx0.min() - m0, 0), min(kx0.max() + m0, img.shape[1])
    Image.fromarray(img[r0:r1, c0:c1].astype(np.uint8)).save(base + "_canvas.png")
    tl, br = to_mm(c0, r0), to_mm(c1, r1)
    doc.layers.add("CANVAS_FRAME")   # snap the _canvas.png corners to this frame in Fusion
    msp.add_lwpolyline([tl, (br[0], tl[1]), br, (tl[0], br[1])], close=True,
                       dxfattribs={"layer": "CANVAS_FRAME"})
    doc.layers.add("LANDMARKS")
    for p in (tip, fL, fR):
        if p is not None:
            msp.add_point(to_mm(*p), dxfattribs={"layer": "LANDMARKS"})
    doc.saveas(base + ".dxf")

    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contour", "x_mm", "y_mm"])
        for k, pts in shapes_mm.items():
            for x, y in pts:
                w.writerow([k, round(x, 4), round(y, 4)])
        for x, y in (crack_mm or []):
            w.writerow(["CRACK", round(x, 4), round(y, 4)])
        if slot:
            for x, y in slot[0]:
                w.writerow(["SLOT_CENTERLINE", round(x, 5), round(y, 5)])
        for k, p in (("PAPILLA_TIP", tip), ("FORNIX_L", fL), ("FORNIX_R", fR)):
            if p is not None:
                x, y = to_mm(*p)
                w.writerow([k, round(x, 4), round(y, 4)])

    ky, kx = np.nonzero(kid)
    a2 = um_px ** 2 / 1e6
    m = {
        "name": name, "um_per_px": round(um_px, 3),
        "kidney_width_mm": round((kx.max() - kx.min()) * mm, 3),
        "kidney_height_mm": round((ky.max() - ky.min()) * mm, 3),
        "kidney_area_mm2": round(kid.sum() * a2, 3),
        "canvas_width_mm": round(br[0] - tl[0], 4), "canvas_height_mm": round(tl[1] - br[1], 4),
        "canvas_top_left_mm": [round(tl[0], 4), round(tl[1], 4)],
        "lumen_area_mm2": round(lumen.sum() * a2, 4) if lumen is not None else None,
        "slot_width_um": args.slot_width if slot else None,
        "slot_centerline_length_mm": round(float(np.sum(np.hypot(*np.diff(slot[0], axis=0).T))), 4) if slot else None,
    }
    if tip is not None and fL is not None and fR is not None:
        m["papilla_len_left_mm"] = round((fL[1] - tip[1]) * mm, 3)
        m["papilla_len_right_mm"] = round((fR[1] - tip[1]) * mm, 3)
        m["fornix_spacing_mm"] = round((fR[0] - fL[0]) * mm, 3)
    if hybrid is not None:
        m["lumen_area_mm2"] = round(hybrid["area"], 4)
    elif "PELVIS_LUMEN" in shapes_mm and round_mm > 0:
        m["lumen_area_mm2"] = round(poly_area_mm(shapes_mm["PELVIS_LUMEN"]), 4)
    m["round_um"] = args.round_um
    m["outer_round_um"] = args.outer_round_um if args.outer_round_um is not None else args.round_um
    m["edited"] = edited
    m["reconstruction"] = ("hybrid" if hybrid is not None and slot and args.hybrid else
                           "edited" if edited else "hybrid" if hybrid is not None else "slot" if slot else
                           "crack" if crack else "automatic")
    with open(base + ".json", "w") as f:
        json.dump(m, f, indent=2)

    prev = (img * 0.55 + 115).astype(np.uint8)
    for mask, col in ((kid, (0, 0, 0)), (lumen, (220, 40, 40)), (second, (30, 150, 60))):
        if mask is not None:
            prev[mask ^ morphology.binary_erosion(mask, morphology.disk(3))] = col
    for p in (tip, fL, fR):
        if p is not None:
            c, r = int(p[0]), int(p[1])
            prev[max(r - 7, 0):r + 7, max(c - 7, 0):c + 7] = (30, 90, 220)
    if crack:
        for (c0, r0), (c1, r1) in zip(crack[:-1], crack[1:]):
            rr, cc = draw.line(int(r0), int(c0), int(r1), int(c1))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    prev[np.clip(rr + dr, 0, prev.shape[0] - 1), np.clip(cc + dc, 0, prev.shape[1] - 1)] = (240, 140, 0)
    if slot_px:
        cl = slot[0]
        cpx = [((x / mm) + origin[0], origin[1] - (y / mm)) for x, y in cl]
        for (c0, r0), (c1, r1) in zip(cpx[:-1], cpx[1:]):
            rr, cc = draw.line(int(r0), int(c0), int(r1), int(c1))
            ok = (rr >= 0) & (rr < prev.shape[0]) & (cc >= 0) & (cc < prev.shape[1])
            prev[rr[ok], cc[ok]] = (220, 40, 40)
        for c0, r0 in slot_px:
            prev[max(int(r0) - 4, 0):int(r0) + 4, max(int(c0) - 4, 0):int(c0) + 4] = (240, 140, 0)
    if edited:
        op_ = [((x / mm) + origin[0], origin[1] - (y / mm)) for x, y in shapes_mm["KIDNEY_OUTER"]]
        for (c0, r0), (c1, r1) in zip(op_, op_[1:] + op_[:1]):
            rr, cc = draw.line(int(r0), int(c0), int(r1), int(c1))
            ok = (rr >= 0) & (rr < prev.shape[0]) & (cc >= 0) & (cc < prev.shape[1])
            prev[rr[ok], cc[ok]] = (0, 90, 220)
    if hybrid is not None:
        hp = [((x / mm) + origin[0], origin[1] - (y / mm)) for x, y in hybrid["pts"]]
        for (c0, r0), (c1, r1) in zip(hp, hp[1:] + hp[:1]):
            rr, cc = draw.line(int(r0), int(c0), int(r1), int(c1))
            ok = (rr >= 0) & (rr < prev.shape[0]) & (cc >= 0) & (cc < prev.shape[1])
            prev[rr[ok], cc[ok]] = (200, 0, 200)
    for r0, c0, r1, c1 in cuts:
        rr, cc = draw.line(r0, c0, r1, c1)
        ok = (rr >= 0) & (rr < prev.shape[0]) & (cc >= 0) & (cc < prev.shape[1])
        prev[rr[ok], cc[ok]] = (0, 160, 0)
    for r0, c0, r1, c1 in seals:
        rr, cc = draw.line(r0, c0, r1, c1)
        ok = (rr >= 0) & (rr < prev.shape[0]) & (cc >= 0) & (cc < prev.shape[1])
        prev[rr[ok], cc[ok]] = (30, 90, 220)
    Image.fromarray(prev).save(base + "_overlay.png")
    print(json.dumps(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--panel", default="UUO_POD3", help="panel key, or 'all'")
    ap.add_argument("--center", nargs=2, type=float, help="custom panel center (pt)")
    ap.add_argument("--name", default="panel")
    ap.add_argument("--bar-um", type=float, default=1180.0, help="scale bar length (um)")
    ap.add_argument("--bar-pt", type=float, help="override scale bar length (pt)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--half", type=float, default=85, help="half crop size (pt)")
    ap.add_argument("--thresh", type=float, default=690, help="RGB-sum tissue threshold")
    ap.add_argument("--min-obj", type=int, default=2000)
    ap.add_argument("--open-r", type=int, default=20, help="opening radius to split neighbors")
    ap.add_argument("--cut-col", type=int, nargs="*", help="manual vertical cut(s), px")
    ap.add_argument("--cut-row", type=int, nargs="*", help="manual horizontal cut(s), px")
    ap.add_argument("--lumen-mode", choices=["hole", "hull"], default="hole")
    ap.add_argument("--lumen-seed", nargs="+", type=float,
                    help="point(s) X Y (pt) inside the pelvic cavity; in hybrid mode give one point "
                         "inside each open part to merge")
    ap.add_argument("--pick-seed", action="store_true",
                    help="click the lumen seed point(s) on the image instead of typing coordinates")
    ap.add_argument("--keep-second", action="store_true", help="also export 2nd cavity")
    ap.add_argument("--seal", nargs=4, type=float, action="append", metavar=("X1", "Y1", "X2", "Y2"),
                    help="seal line (pt) across an opening, e.g. the hilum mouth; repeatable")
    ap.add_argument("--smooth", type=float, default=0,
                    help="Gaussian smoothing (px) before thresholding; try 3 with --thresh 620")
    ap.add_argument("--edit", action="store_true",
                    help="open an editor before export: erase/add parts and connect the lumen "
                         "to the next open part; edits are saved to <name>_edits.json")
    ap.add_argument("--edits", help="apply edits saved earlier (<name>_edits.json)")
    ap.add_argument("--cut", nargs=4, type=float, action="append", metavar=("X1", "Y1", "X2", "Y2"),
                    help="cut line (pt) separating this section from a touching neighbour; repeatable")
    ap.add_argument("--pick-cut", action="store_true",
                    help="click cut lines on the image (pairs of points); saved for reuse")
    ap.add_argument("--seal-extend", type=float, default=15,
                    help="extend each seal line past its clicked ends by this many px (default 15)")
    ap.add_argument("--pick-seal", action="store_true",
                    help="click seal lines on the image (pairs of points); saved for reuse")
    ap.add_argument("--close-r", type=int, default=0,
                    help="closing radius (px) to seal tiny breaks so cavities are closed")
    ap.add_argument("--slot-pts", nargs="+", type=float, metavar="XY",
                    help="centerline points X1 Y1 X2 Y2 ... (pt), fornix to fornix around the papilla tip")
    ap.add_argument("--slot-width", type=float, default=8.0, help="slot width (um), default 8")
    ap.add_argument("--hybrid", action="store_true",
                    help="merge the automatic (open) cavity with the slot from the centerline")
    ap.add_argument("--round-um", type=float, default=0,
                    help="round all outlines: remove features narrower than 2R and round corners "
                         "to radius >= R (um); also the fornix rounding for sensitivity. 0 = off")
    ap.add_argument("--outer-round-um", type=float, default=None,
                    help="rounding radius (um) for the kidney outer outline only; use a larger value "
                         "(e.g. 30-50) to remove sectioning tears and notches; default = --round-um")
    ap.add_argument("--min-wall-um", type=float, default=20,
                    help="tissue kept between a cavity and the outer surface at seal lines "
                         "(um, default 20): the seal becomes a thin wall")
    ap.add_argument("--outline-sigma", type=float, default=4,
                    help="Gaussian smoothing (px) of the kidney mask before contouring (default 4)")
    ap.add_argument("--lumen-sigma", type=float, default=2.5,
                    help="Gaussian smoothing (px) of the cavity mask before contouring (default 2.5)")
    ap.add_argument("--min-cavity-um2", type=float, default=0,
                    help="smallest open cavity (um^2) merged in hybrid mode; default ~1000 um^2")
    ap.add_argument("--junction-r", type=float, default=0,
                    help="fillet radius (um) at hybrid junctions; default = half slot width")
    ap.add_argument("--pick-slot", action="store_true",
                    help="click the slot centerline on the image; points are saved for reuse")
    ap.add_argument("--crack-seed", nargs=2, type=float,
                    help="point (pt) inside a thin slit: export its centerline as a crack")
    ap.add_argument("--out", default="outlines")
    args = ap.parse_args()

    pdf_document = pymupdf.open(args.pdf)
    page = pdf_document[0]
    bar_pt = args.bar_pt or find_scale_bar_pt(page)
    um_px = args.bar_um / bar_pt / (args.dpi / 72)
    print(f"scale bar {bar_pt:.2f} pt -> {um_px:.3f} um/px")

    if (args.slot_pts or args.pick_slot or args.pick_seal or args.pick_cut or args.cut or args.edit
            or args.edits or args.pick_seed) and args.panel == "all" and not args.center:
        raise SystemExit("Slot points belong to one section; use --panel <name> or --center.")
    if args.center:
        process(page, args.name, tuple(args.center), args, um_px)
    elif args.panel == "all":
        for k, c in PANELS.items():
            process(page, k, c, args, um_px)
    else:
        process(page, args.panel, PANELS[args.panel], args, um_px)



def launch_gui():
    """No arguments: launch a desktop form. CLI arguments still work as before."""
    import sys
    import subprocess
    import threading
    import queue
    from pathlib import Path
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("Kidney Outline Extraction")
    h_ = min(1230, root.winfo_screenheight() - 90)   # fit smaller screens
    root.geometry(f"860x{h_}+40+20")
    root.minsize(720, 650)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    outer_ = ttk.Frame(root)
    outer_.grid(sticky="nsew")
    outer_.columnconfigure(0, weight=1)
    outer_.rowconfigure(0, weight=1)
    canvas_ = tk.Canvas(outer_, highlightthickness=0)
    vbar_ = ttk.Scrollbar(outer_, orient="vertical", command=canvas_.yview)
    canvas_.configure(yscrollcommand=vbar_.set)
    canvas_.grid(row=0, column=0, sticky="nsew")
    vbar_.grid(row=0, column=1, sticky="ns")
    box = ttk.Frame(canvas_, padding=14)
    win_ = canvas_.create_window((0, 0), window=box, anchor="nw")
    box.bind("<Configure>", lambda e: canvas_.configure(scrollregion=canvas_.bbox("all")))
    canvas_.bind("<Configure>", lambda e: canvas_.itemconfigure(win_, width=e.width))
    canvas_.bind_all("<MouseWheel>", lambda e: canvas_.yview_scroll(int(-e.delta / 120), "units"))
    box.columnconfigure(1, weight=1)
    fields = {}

    def entry(row, label, key, default=""):
        ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", pady=4)
        var = tk.StringVar(value=default)
        ttk.Entry(box, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8)
        fields[key] = var
        return var

    pdf = entry(0, "PDF file", "pdf")
    out = entry(1, "Output folder", "out")
    def browse_pdf():
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if path:
            pdf.set(path)
            out.set(str(Path(path).parent / "outlines"))
    def browse_out():
        path = filedialog.askdirectory()
        if path:
            out.set(path)
    ttk.Button(box, text="Browse PDF...", command=browse_pdf).grid(row=0, column=2)
    ttk.Button(box, text="Browse folder...", command=browse_out).grid(row=1, column=2)
    ttk.Label(box, text="Panel").grid(row=2, column=0, sticky="w")
    panel = tk.StringVar(value="UUO_POD3")
    ttk.Combobox(box, textvariable=panel, values=list(PANELS) + ["all"],
                 state="readonly").grid(row=2, column=1, sticky="ew", padx=8, pady=5)
    presets = tk.BooleanVar(value=True)
    ttk.Checkbutton(box, text="Use recommended presets: POD3 cut-col=1120; POD7 hull (600 DPI)",
                    variable=presets).grid(row=3, column=0, columnspan=3, sticky="w")
    ttk.Label(box, text="Preset coordinates match the original H&E PDF layout only. For other PDFs, enter a custom center.",
              wraplength=780).grid(row=4, column=0, columnspan=3, sticky="w", pady=6)
    mode = entry(5, "Lumen mode (hole / hull)", "lumen-mode", "hole")
    entry(6, "Vertical cuts (pixels, optional)", "cut-col")
    entry(7, "Horizontal cuts (pixels, optional)", "cut-row")
    entry(8, "Scale bar length (um, verify)", "bar-um", "1180")
    entry(9, "Scale bar PDF length (pt, auto if blank)", "bar-pt")
    entry(10, "Render DPI", "dpi", "600")
    entry(11, "Crop half-width (pt)", "half", "85")
    entry(12, "Tissue threshold / min object / opening radius", "segmentation", "690 2000 20")
    entry(13, "Custom center X Y (PDF pt, optional)", "center")
    entry(14, "Custom panel name", "name", "panel")
    entry(15, "Lumen seed(s) X Y ... (PDF pt; hybrid: one per open part)", "lumen-seed")
    keep = tk.BooleanVar(value=False)
    ttk.Checkbutton(box, text="Export second cavity", variable=keep).grid(row=16, column=0, sticky="w")
    pick_seed = tk.BooleanVar(value=False)
    ttk.Checkbutton(box, text="Pick lumen seed(s) by clicking (one panel only)",
                    variable=pick_seed).grid(row=16, column=1, sticky="w")
    entry(17, "Seal lines X1 Y1 X2 Y2 (pt; separate lines with ;)", "seal")
    entry(18, "Crack seed X Y (PDF pt, optional)", "crack-seed")
    entry(19, "Closing radius (px, 0 = off)", "close-r", "0")
    entry(20, "Smoothing sigma (px, 0 = off; try 3 with threshold 620)", "smooth", "0")
    entry(21, "Slot centerline points X1 Y1 X2 Y2 ... (PDF pt)", "slot-pts")
    entry(22, "Slot width (um)", "slot-width", "8")
    pick = tk.BooleanVar(value=False)
    ttk.Checkbutton(box, text="Pick slot points by clicking on the image (one panel only)",
                    variable=pick).grid(row=23, column=0, columnspan=3, sticky="w")
    entry(26, "Round outlines R (um) | outer R (um, blank = same; e.g. 40)", "round-um", "0")
    hyb = tk.BooleanVar(value=False)
    ttk.Checkbutton(box, text="Hybrid: merge the automatic cavity with the slot (use a lumen seed + slot points)",
                    variable=hyb).grid(row=25, column=0, columnspan=3, sticky="w")
    do_edit = tk.BooleanVar(value=False)
    ttk.Checkbutton(box, text="Edit outlines before export (erase / add / connect; one panel only)",
                    variable=do_edit).grid(row=31, column=0, columnspan=3, sticky="w")
    entry(32, "Edits file to re-apply (optional, <name>_edits.json)", "edits")
    entry(29, "Minimum wall at seal lines (um; 100 recommended)", "min-wall", "100")
    pick_cut = tk.BooleanVar(value=False)
    ttk.Checkbutton(box, text="Pick cut lines by clicking (separate a touching neighbour section; one panel only)",
                    variable=pick_cut).grid(row=27, column=0, columnspan=3, sticky="w")
    entry(28, "Cut lines X1 Y1 X2 Y2 (pt; separate lines with ;)", "cut")
    pick_seal = tk.BooleanVar(value=False)
    ttk.Checkbutton(box, text="Pick seal lines by clicking (pairs of points, e.g. across the hilum mouth; one panel only)",
                    variable=pick_seal).grid(row=24, column=0, columnspan=3, sticky="w")
    log = tk.Text(box, height=8, wrap="word")
    log.grid(row=34, column=0, columnspan=3, sticky="nsew", pady=8)
    box.rowconfigure(34, weight=1)
    events = queue.Queue()
    state = {"busy": False}

    def run():
        if state["busy"]:
            return
        try:
            source = Path(pdf.get()).expanduser().resolve()
            if not source.is_file() or source.suffix.lower() != ".pdf":
                raise ValueError("Please select an existing PDF file.")
            destination = Path(out.get()).expanduser().resolve() if out.get().strip() else source.parent / "outlines"
            dpi = int(fields["dpi"].get())
            if dpi <= 0 or float(fields["bar-um"].get()) <= 0 or float(fields["half"].get()) <= 0:
                raise ValueError("DPI, scale bar length and crop half-width must be positive.")
            if mode.get() not in ("hole", "hull"):
                raise ValueError("Lumen mode must be hole or hull.")
            if presets.get() and dpi != 600:
                raise ValueError("Cut presets require 600 DPI. Set DPI to 600, or disable presets and adjust cuts manually.")
            options = []
            for group in [g.split() for g in fields["seal"].get().split(";") if g.strip()]:
                if len(group) != 4:
                    raise ValueError("Each seal line needs four numbers: X1 Y1 X2 Y2.")
                [float(v) for v in group]
                options += ["--seal"] + group
            if int(fields["close-r"].get() or 0) < 0:
                raise ValueError("Closing radius must be zero or positive.")
            options += ["--close-r", str(int(fields["close-r"].get() or 0))]
            if float(fields["smooth"].get() or 0) < 0:
                raise ValueError("Smoothing sigma must be zero or positive.")
            options += ["--smooth", str(float(fields["smooth"].get() or 0))]
            sp = fields["slot-pts"].get().split()
            if sp:
                if len(sp) % 2 or len(sp) < 6:
                    raise ValueError("Slot points need at least three X Y pairs.")
                [float(v) for v in sp]
                options += ["--slot-pts"] + sp
            if float(fields["slot-width"].get() or 8) <= 0:
                raise ValueError("Slot width must be positive.")
            options += ["--slot-width", str(float(fields["slot-width"].get() or 8))]
            if (sp or pick.get()) and panel.get() == "all" and not fields["center"].get().strip():
                raise ValueError("Slot points belong to one section; choose a single panel.")
            if hyb.get():
                if not (sp or pick.get()):
                    raise ValueError("Hybrid needs slot points: enter them or tick 'Pick slot points'.")
                options += ["--hybrid"]
            rv = (fields["round-um"].get() or "0").replace("|", " ").replace(",", " ").split()
            if len(rv) not in (1, 2) or any(float(v) < 0 for v in rv):
                raise ValueError("Rounding: give R, or R and the outer R (e.g. '5 40'), both >= 0.")
            options += ["--round-um", str(float(rv[0]))]
            if len(rv) == 2:
                options += ["--outer-round-um", str(float(rv[1]))]
            for group in [g.split() for g in fields["cut"].get().split(";") if g.strip()]:
                if len(group) != 4:
                    raise ValueError("Each cut line needs four numbers: X1 Y1 X2 Y2.")
                [float(v) for v in group]
                options += ["--cut"] + group
            mw = float(fields["min-wall"].get() or 20)
            if mw < 0:
                raise ValueError("Minimum wall must be zero or positive.")
            options += ["--min-wall-um", str(mw)]
            if fields["edits"].get().strip():
                options += ["--edits", fields["edits"].get().strip()]
            if do_edit.get():
                if panel.get() == "all" and not fields["center"].get().strip():
                    raise ValueError("Editing works on one section; choose a single panel.")
                options += ["--edit"]
            if pick_cut.get():
                if panel.get() == "all" and not fields["center"].get().strip():
                    raise ValueError("Cut picking works on one section; choose a single panel.")
                options += ["--pick-cut"]
            if pick_seed.get():
                if panel.get() == "all" and not fields["center"].get().strip():
                    raise ValueError("Seed picking works on one section; choose a single panel.")
                options += ["--pick-seed"]
            if pick_seal.get():
                if panel.get() == "all" and not fields["center"].get().strip():
                    raise ValueError("Seal picking works on one section; choose a single panel.")
                options += ["--pick-seal"]
            if pick.get():
                options += ["--pick-slot"]
            for key in ("bar-um", "bar-pt", "dpi", "half", "center", "name", "lumen-seed", "crack-seed", "cut-col", "cut-row"):
                value = fields[key].get().strip()
                if not value:
                    continue
                values = value.split() if key in ("center", "lumen-seed", "crack-seed", "cut-col", "cut-row") else [value]
                if key in ("center", "crack-seed") and len(values) != 2:
                    raise ValueError(key + " requires two coordinates.")
                if key == "lumen-seed" and len(values) % 2:
                    raise ValueError("Lumen seed needs X Y pairs (one pair per open part in hybrid mode).")
                if key != "name":
                    for v in values:
                        (int if key in ("dpi", "cut-col", "cut-row") else float)(v)
                options += ["--" + key] + values
            name = fields["name"].get().strip()
            if fields["center"].get().strip() and (not name or Path(name).name != name or name in (".", "..") or any(c in name for c in '<>:"/\\|?*')):
                raise ValueError("Use a plain filename for the custom name, without paths or special characters.")
            seg = fields["segmentation"].get().split()
            if len(seg) != 3 or float(seg[0]) <= 0 or int(seg[1]) < 0 or int(seg[2]) < 0:
                raise ValueError("Enter valid threshold, minimum object size and opening radius values.")
            for key, value in zip(("thresh", "min-obj", "open-r"), seg):
                options += ["--" + key, value]
            if keep.get():
                options += ["--keep-second"]
            keys = list(PANELS) if panel.get() == "all" and not fields["center"].get().strip() else [panel.get()]
            commands = []
            for key in keys:
                extra = ["--lumen-mode", mode.get()]
                if presets.get() and not fields["center"].get().strip():
                    if key == "UUO_POD3" and not fields["cut-col"].get().strip():
                        extra += ["--cut-col", "1120"]
                    if key == "UUO_POD7":
                        extra = ["--lumen-mode", "hull"]
                commands.append([sys.executable, "-u", str(Path(__file__).resolve()), str(source),
                                 "--panel", key, "--out", str(destination)] + options + extra)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        if destination.exists() and any(destination.glob("*.dxf")):
            if not messagebox.askyesno("Existing results", "The output folder already contains DXF files. Results with the same names will be overwritten. Continue?"):
                return
        log.delete("1.0", "end")
        state["busy"] = True
        button.configure(state="disabled")
        def worker():
            try:
                for command in commands:
                    events.put(("log", "Processing " + command[command.index("--panel") + 1] + "\n"))
                    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, encoding="utf-8", errors="replace",
                                          env=dict(os.environ, PYTHONIOENCODING="utf-8"), **kwargs) as proc:
                        for line in proc.stdout:
                            events.put(("log", line))
                        if proc.wait() != 0:
                            raise RuntimeError("Processing failed. See the error details in the log.")
                events.put(("done", str(destination)))
            except Exception as exc:
                events.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    button = ttk.Button(box, text="Start extraction", command=run)
    button.grid(row=33, column=0, columnspan=3, pady=7)
    def poll():
        try:
            while True:
                kind, value = events.get_nowait()
                log.insert("end", value + ("\n" if kind != "log" else ""))
                log.see("end")
                if kind in ("done", "error"):
                    state["busy"] = False
                    button.configure(state="normal")
                    if kind == "done":
                        messagebox.showinfo("Extraction complete", "Results saved to:\n" + value + "\n\nCheck the overlay PNG files to verify contours and scale.")
                    else:
                        messagebox.showerror("Extraction failed", value)
        except queue.Empty:
            pass
        root.after(100, poll)
    def close():
        if state["busy"]:
            messagebox.showinfo("Processing", "Please wait for extraction to finish before closing the window.")
        else:
            root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    poll()
    root.mainloop()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        try:
            launch_gui()
        except Exception:   # keep the window open so the error can be read
            import traceback
            traceback.print_exc()
            input("\nThe program stopped with the error above. Press Enter to close...")
    else:
        main()
