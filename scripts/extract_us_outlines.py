"""
Extract the pelvic lumen and kidney outline from a B-mode ultrasound frame (e.g. Vevo 2100
screenshots embedded in a PDF) and export them in the same format as
extract_kidney_outlines.py, so the geometry can be solved with batch_fea.py.

Requirements:
    pip install pymupdf scikit-image scipy numpy pillow ezdxf matplotlib shapely

Typical use
    1. List the frames in a PDF and save thumbnails:
         python extract_us_outlines.py --pdf UUO_serial_US.pdf --list --out us
    2. Segment one frame (click the lumen seed and the kidney outline):
         python extract_us_outlines.py --pdf UUO_serial_US.pdf --xref 11 --name US_UUO_POD3 \\
             --fov-width-mm 14.08 --pick-seed --pick-kidney --out us
       or re-run exactly with saved coordinates (pixels of the frame):
         python extract_us_outlines.py --pdf UUO_serial_US.pdf --xref 11 --name US_UUO_POD3 \\
             --fov-width-mm 14.08 --seed 940 530 --kidney-pts x1 y1 x2 y2 ... --out us
    3. Check and solve with batch_fea.py:
         python batch_fea.py us/US_UUO_POD3 --dry-run --out check

Scale
    --fov-width-mm W   width of the B-mode field of view printed on the screen ("Width");
                       the field of view is detected automatically and mm/px = W / width_px.
    --mm-per-px S      give the scale directly instead.
    The spacing of the depth-ruler ticks is printed as an independent check.

Lumen (anechoic pelvic space)
    The frame is despeckled (median + Gaussian). The lumen is the dark region connected to
    the seed point(s). With --thresh auto (default) the region is grown over a range of
    thresholds above the seed intensity and the threshold just before the largest jump in
    area (the leak into surrounding dark tissue) is used; the area-threshold curve is saved. --barrier X1 Y1 X2 Y2 draws a bright line to block a leak (repeatable).

Kidney outline
    Ultrasound rarely shows a closed, high-contrast capsule, so the outline is traced by
    hand (--pick-kidney, or --kidney-pts) and smoothed with a periodic spline.

Coordinates: output in mm, origin at the papilla tip (or the lumen centroid if no tip is
found), x to the right, y up (image depth increases downward on screen).
"""
import argparse, csv, json, os, warnings
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology, draw
import ezdxf

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ input -----------
def list_frames(pdf, out):
    import pymupdf
    doc = pymupdf.open(pdf)
    os.makedirs(out, exist_ok=True)
    rows = []
    for pno, page in enumerate(doc):
        for info in page.get_image_info(xrefs=True):
            if info["width"] < 300:
                continue
            x0, y0, x1, y1 = info["bbox"]
            rows.append((pno, info["xref"], info["width"], info["height"], x0, y0))
            img = load_frame(pdf, info["xref"])
            Image.fromarray(img).resize((info["width"] // 4, info["height"] // 4)).save(
                os.path.join(out, f"frame_xref{info['xref']}.png"))
    rows.sort(key=lambda r: (r[0], round(r[5] / 20), r[4]))   # page, row, column
    print("page  xref  size        position on page (x, y)")
    for pno, xref, w, h, x0, y0 in rows:
        print(f"{pno:4d}  {xref:4d}  {w}x{h}   ({x0:.0f}, {y0:.0f})")
    print(f"thumbnails saved in {out}/frame_xref*.png")


def load_frame(pdf, xref):
    import pymupdf
    doc = pymupdf.open(pdf)
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n - pix.alpha > 3 or pix.n < 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return a[..., :3].copy()


# ------------------------------------------------------------------ scale -----------
def detect_fov(gray):
    """Bounding box (r0, r1, c0, c1) of the B-mode field of view: the widest run of
    columns, and tallest run of rows, with mean intensity above background."""
    def longest_run(mask):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return None
        runs = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
        r = max(runs, key=len)
        return int(r[0]), int(r[-1]) + 1
    h, w = gray.shape
    band = gray[int(0.3 * h):int(0.7 * h)]
    c = longest_run(band.mean(0) > 15)
    rows = gray[:, c[0]:c[1]].mean(1) > 15 if c else gray.mean(1) > 15
    r = longest_run(rows)
    return r[0], r[1], c[0], c[1]


def ruler_spacing(gray, c1):
    """Median spacing (px) of the depth-ruler ticks just right of the field of view."""
    best = None
    for x in range(c1 + 2, min(c1 + 40, gray.shape[1])):
        col = gray[:, x] > 150
        lab, n = ndi.label(col)
        if 8 <= n <= 60:
            cen = np.array(ndi.center_of_mass(col, lab, range(1, n + 1)))[:, 0]
            d = np.diff(np.sort(cen))
            d = d[d > 5]
            if len(d) >= 5:
                s = float(np.median(d))
                if best is None or n > best[1]:
                    best = (s, n)
    return best[0] if best else None


# ------------------------------------------------------------------ segmentation ----
def segment_lumen(smooth, fov, seeds, thresh, barriers, open_r, close_r):
    r0, r1, c0, c1 = fov
    inside = np.zeros_like(smooth, bool)
    inside[r0:r1, c0:c1] = True
    img = smooth.copy()
    for (xa, ya, xb, yb) in barriers:                # bright lines block leaks
        rr, cc = draw.line(int(ya), int(xa), int(yb), int(xb))
        line = np.zeros_like(inside)
        ok = (rr >= 0) & (rr < line.shape[0]) & (cc >= 0) & (cc < line.shape[1])
        line[rr[ok], cc[ok]] = True
        img[morphology.binary_dilation(line, morphology.disk(2))] = 255

    def region(t):
        dark = (img < t) & inside
        if open_r:
            dark = morphology.binary_opening(dark, morphology.disk(open_r))
        lab, _ = ndi.label(dark)
        ids = {lab[int(y), int(x)] for x, y in seeds} - {0}
        if not ids:
            return np.zeros_like(dark)
        m = np.isin(lab, list(ids))
        if close_r:
            m = morphology.binary_closing(m, morphology.disk(close_r))
        return ndi.binary_fill_holes(m)

    if thresh != "auto":
        t = float(thresh)
        return region(t), t, None
    base = max(float(np.mean([img[int(y), int(x)] for x, y in seeds])), 1.0)
    ts = np.arange(base + 4, base + 90, 2.0)
    areas = np.array([region(t).sum() for t in ts], float)
    # the leak is the largest relative jump in area; take the threshold just before it
    ok = areas[:-1] > 500
    ratio = np.where(ok, areas[1:] / np.maximum(areas[:-1], 1), 0)
    i = int(np.argmax(ratio)) if ratio.max() > 1.3 else len(ts) - 1
    chosen = ts[i]
    return region(chosen), chosen, list(zip(ts.tolist(), areas.tolist()))


def landmarks(lumen, tissue):
    """Papilla tip and fornices, as in extract_kidney_outlines.py: the papilla is the tissue
    enclosed by the convex hull of the lumen; tip = its point deepest into the lumen,
    fornices = the two ends of the lumen on either side of the papilla."""
    try:
        hull = morphology.convex_hull_image(lumen)
        pap = hull & ~lumen & tissue
        pl, n = ndi.label(pap)
        if n == 0:
            return None, None, None
        pap = pl == np.bincount(pl.ravel())[1:].argmax() + 1
        # papilla axis: from its base (hull edge) to the farthest papilla point
        edge = morphology.binary_dilation(~hull, morphology.disk(1)) & pap
        py, px = np.nonzero(pap)
        if edge.any():
            ey, ex = np.nonzero(edge)
            base = np.array([ex.mean(), ey.mean()])
        else:
            base = np.array([px.mean(), py.max()])
        d = np.hypot(px - base[0], py - base[1])
        k = d.argmax()
        tip = (float(px[k]), float(py[k]))
        # fornices: lumen pixels closest to the papilla base on each side of the axis
        ly, lx = np.nonzero(lumen)
        ax = np.array(tip) - base
        ax /= np.linalg.norm(ax)
        side = (lx - base[0]) * ax[1] - (ly - base[1]) * ax[0]
        proj = (lx - base[0]) * ax[0] + (ly - base[1]) * ax[1]
        out = []
        for sgn in (-1, 1):
            msk = np.sign(side) == sgn
            if not msk.any():
                out.append(None)
                continue
            j = np.argmin(np.where(msk, proj, np.inf))
            out.append((float(lx[j]), float(ly[j])))
        return tip, out[0], out[1]
    except Exception:
        return None, None, None


# ------------------------------------------------------------------ geometry --------
def smooth_closed(pts, n=400):
    from scipy.interpolate import splprep, splev
    p = np.asarray(pts, float)
    p = p[np.r_[True, np.hypot(*np.diff(p, axis=0).T) > 1e-9]]
    if np.hypot(*(p[0] - p[-1])) < 1e-9:
        p = p[:-1]
    tck, _ = splprep([p[:, 0], p[:, 1]], s=len(p) * 2.0, per=1, k=3)
    x, y = splev(np.linspace(0, 1, n, endpoint=False), tck)
    return np.c_[x, y]


def mask_contour(mask, sigma=1.5, tol=1.0):
    sm = ndi.gaussian_filter(mask.astype(float), sigma)
    c = max(measure.find_contours(sm, 0.5), key=len)
    c = measure.approximate_polygon(c, tol)[:-1]
    return np.c_[c[:, 1], c[:, 0]]                    # (x, y) pixels


def round_poly(pts_mm, r_mm):
    from shapely.geometry import Polygon
    g = Polygon(pts_mm).buffer(0)
    g = g.buffer(-r_mm, quad_segs=16).buffer(r_mm, quad_segs=16)
    g = g.buffer(r_mm, quad_segs=16).buffer(-r_mm, quad_segs=16)
    if g.is_empty:
        return pts_mm
    if g.geom_type != "Polygon":
        g = max(g.geoms, key=lambda q: q.area)
    return np.array(g.exterior.coords)[:-1]


# ------------------------------------------------------------------ picking --------
def pick(img, title, closed=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(img, cmap="gray")
    ax.set_title(title + "\nLeft-click adds a point, right-click removes the last one, "
                 "Enter finishes (turn zoom/pan off before clicking).", fontsize=9)
    pts = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=None)
    plt.close(fig)
    return [(float(x), float(y)) for x, y in pts]


# ------------------------------------------------------------------ main ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="PDF with embedded ultrasound frames")
    ap.add_argument("--xref", type=int, help="image xref in the PDF (see --list)")
    ap.add_argument("--image", help="or an image file (png/jpg/tif) instead of --pdf/--xref")
    ap.add_argument("--list", action="store_true", help="list the frames in --pdf and exit")
    ap.add_argument("--name", default="US_frame")
    ap.add_argument("--out", default="us")
    ap.add_argument("--fov-width-mm", type=float, help="B-mode field-of-view width (mm)")
    ap.add_argument("--mm-per-px", type=float, help="scale, if not using --fov-width-mm")
    ap.add_argument("--median", type=int, default=7, help="median filter size (px)")
    ap.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing (px)")
    ap.add_argument("--thresh", default="auto", help="'auto' or a fixed intensity (0-255)")
    ap.add_argument("--seed", nargs="+", type=float, help="lumen seed(s) X Y ... (frame px)")
    ap.add_argument("--pick-seed", action="store_true")
    ap.add_argument("--barrier", nargs=4, type=float, action="append",
                    metavar=("X1", "Y1", "X2", "Y2"), help="line (px) that blocks a lumen leak")
    ap.add_argument("--open-r", type=int, default=2)
    ap.add_argument("--close-r", type=int, default=3)
    ap.add_argument("--kidney-pts", nargs="+", type=float, help="kidney outline X Y ... (px)")
    ap.add_argument("--pick-kidney", action="store_true")
    ap.add_argument("--round-um", type=float, default=10, help="lumen rounding radius (um)")
    a = ap.parse_args()

    if a.list:
        list_frames(a.pdf, a.out)
        return
    if a.image:
        img = np.asarray(Image.open(a.image).convert("RGB"))
    else:
        img = load_frame(a.pdf, a.xref)
    gray = img.astype(float).mean(2)
    fov = detect_fov(gray)
    r0, r1, c0, c1 = fov
    spacing = ruler_spacing(gray, c1)
    if a.mm_per_px:
        mmpx = a.mm_per_px
    elif a.fov_width_mm:
        mmpx = a.fov_width_mm / (c1 - c0)
    else:
        raise SystemExit("Give --fov-width-mm (the 'Width' on the screen) or --mm-per-px.")
    print(f"field of view: x {c0}-{c1}, y {r0}-{r1} px;  scale {mmpx * 1000:.2f} um/px")
    if spacing:
        print(f"  check: depth-ruler tick spacing {spacing:.1f} px = {spacing * mmpx:.3f} mm "
              "(should equal the ruler's tick interval, e.g. 0.5 mm)")

    smooth = ndi.gaussian_filter(ndi.median_filter(gray, a.median), a.sigma)
    os.makedirs(a.out, exist_ok=True)
    rec = {"name": a.name, "source": a.image or f"{a.pdf} xref {a.xref}"}

    if a.pick_seed:
        a.seed = [v for p in pick(smooth, a.name + ": click inside the dark pelvic space") for v in p]
    if not a.seed:
        raise SystemExit("Give --seed X Y (or --pick-seed).")
    seeds = list(zip(a.seed[0::2], a.seed[1::2]))
    lumen, t, curve = segment_lumen(smooth, fov, seeds, a.thresh, a.barrier or [],
                                    a.open_r, a.close_r)
    if lumen.sum() == 0:
        raise SystemExit("Seed is not in a dark region; check the seed or use a higher --thresh.")
    print(f"  lumen threshold {t:.1f} ({'auto' if a.thresh == 'auto' else 'fixed'}), "
          f"area {lumen.sum() * mmpx ** 2:.3f} mm^2")
    if lumen[r0 + 2:r1 - 2, c0 + 2:c1 - 2].sum() < lumen.sum():
        print("  WARNING: the lumen touches the edge of the field of view - add a --barrier "
              "or lower --thresh")

    if a.pick_kidney:
        k = pick(img, a.name + ": click around the kidney outline (any number of points)")
        a.kidney_pts = [v for p in k for v in p]
    kid_px = None
    if a.kidney_pts and len(a.kidney_pts) >= 12:
        kid_px = smooth_closed(list(zip(a.kidney_pts[0::2], a.kidney_pts[1::2])))
    else:
        print("  note: no kidney outline given (need >= 6 points); exporting the lumen only")

    tissue = np.ones_like(lumen)
    if kid_px is not None:
        from matplotlib.path import Path
        yy, xx = np.mgrid[0:lumen.shape[0], 0:lumen.shape[1]]
        tissue = Path(kid_px).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(lumen.shape)
        tissue &= ~lumen
        if not (lumen & ~tissue & ~Path(kid_px).contains_points(
                np.c_[xx.ravel(), yy.ravel()]).reshape(lumen.shape)).sum() == 0:
            print("  WARNING: part of the lumen lies outside the traced kidney outline")
    tip, fL, fR = landmarks(lumen, tissue)
    lum_px = mask_contour(lumen)
    origin = tip if tip else (float(lum_px[:, 0].mean()), float(lum_px[:, 1].mean()))
    to_mm = lambda x, y: ((x - origin[0]) * mmpx, -(y - origin[1]) * mmpx)
    lum_mm = np.array([to_mm(*p) for p in lum_px])
    if a.round_um > 0:
        lum_mm = round_poly(lum_mm, a.round_um / 1000.0)
    shapes = {"PELVIS_LUMEN": lum_mm}
    if kid_px is not None:
        shapes["KIDNEY_OUTER"] = np.array([to_mm(*p) for p in kid_px])

    # metrics: lumen area, AP diameter (screen depth direction) and transverse diameter
    ly, lx = np.nonzero(lumen)
    m = {"name": a.name, "um_per_px": round(mmpx * 1000, 3), "threshold": round(t, 1),
         "lumen_area_mm2": round(lumen.sum() * mmpx ** 2, 4),
         "lumen_APD_mm": round((ly.max() - ly.min() + 1) * mmpx, 3),
         "lumen_transverse_mm": round((lx.max() - lx.min() + 1) * mmpx, 3)}
    if kid_px is not None:
        from shapely.geometry import Polygon
        kp = Polygon(shapes["KIDNEY_OUTER"])
        m["kidney_area_mm2"] = round(kp.area, 3)
        m["kidney_width_mm"] = round(float(np.ptp(shapes["KIDNEY_OUTER"][:, 0])), 3)
        m["kidney_height_mm"] = round(float(np.ptp(shapes["KIDNEY_OUTER"][:, 1])), 3)
    if tip and fL and fR:
        m["fornix_spacing_mm"] = round(float(np.hypot(fL[0] - fR[0], fL[1] - fR[1]) * mmpx), 3)

    base = os.path.join(a.out, a.name)
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    for k_, pts in shapes.items():
        doc.layers.add(k_)
        p = [tuple(q) for q in pts]
        msp.add_spline(dxfattribs={"layer": k_}).fit_points = p + [p[0]]
    doc.layers.add("LANDMARKS")
    for p in (tip, fL, fR):
        if p:
            msp.add_point(to_mm(*p), dxfattribs={"layer": "LANDMARKS"})
    doc.saveas(base + ".dxf")
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contour", "x_mm", "y_mm"])
        for k_, pts in shapes.items():
            for x, y in pts:
                w.writerow([k_, round(x, 4), round(y, 4)])
        for k_, p in (("PAPILLA_TIP", tip), ("FORNIX_L", fL), ("FORNIX_R", fR)):
            if p:
                x, y = to_mm(*p)
                w.writerow([k_, round(x, 4), round(y, 4)])
    with open(base + ".json", "w") as f:
        json.dump(m, f, indent=2)
    rec.update({"fov_width_mm": a.fov_width_mm, "mm_per_px": a.mm_per_px, "seed": a.seed,
                "thresh": a.thresh, "barrier": a.barrier, "kidney_pts": a.kidney_pts,
                "median": a.median, "sigma": a.sigma, "round_um": a.round_um})
    rec["threshold_used"] = t
    rec["area_vs_threshold"] = curve
    with open(base + "_inputs.json", "w") as f:
        json.dump(rec, f, indent=1)

    prev = np.stack([gray] * 3, -1).astype(np.uint8)
    e = lumen ^ morphology.binary_erosion(lumen, morphology.disk(2))
    prev[e] = (255, 60, 60)
    if kid_px is not None:
        pts = np.vstack([kid_px, kid_px[:1]])
        for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
            rr, cc = draw.line(int(ya), int(xa), int(yb), int(xb))
            ok = (rr >= 0) & (rr < prev.shape[0]) & (cc >= 0) & (cc < prev.shape[1])
            prev[rr[ok], cc[ok]] = (60, 200, 255)
    for p in (tip, fL, fR):
        if p:
            prev[int(p[1]) - 5:int(p[1]) + 5, int(p[0]) - 5:int(p[0]) + 5] = (255, 220, 0)
    Image.fromarray(prev[r0:r1, c0:c1]).save(base + "_overlay.png")
    print(json.dumps(m))
    print(f"  inputs saved to {a.name}_inputs.json (reuse the seed / kidney points to reproduce)")


if __name__ == "__main__":
    main()
