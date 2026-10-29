# kidney-pelvis-fem

Image-based finite element workflow for the renal pelvis and papilla of the obstructed
murine kidney. It turns calibrated histological sections into plane-strain hyperelastic
models, solves them with Ansys Mechanical APDL through PyMAPDL, and reports
pressure-normalized measures of fornix, papillary and urothelial loading.

This code accompanies:
> [Authors]. [Title]. *Journal of Biomechanics* ([year]). [DOI]

## Contents

| Path | Purpose |
|---|---|
| `scripts/extract_kidney_outlines.py` | Segmentation and geometry reconstruction from a PDF figure (opens a GUI when run without arguments). Writes DXF/CSV/JSON and overlay images. |
| `scripts/extract_us_outlines.py` | Pelvic lumen and kidney outline from B-mode ultrasound frames (scale from the field of view, despeckling, seeded lumen segmentation, traced kidney outline); same output format. |
| `scripts/batch_fea.py` | Builds and solves the plane-strain models in MAPDL; writes summary metrics, boundary profiles and contour maps. `--dry-run`, `--replot` and `--rescore` work without MAPDL. |
| `hpc/submit_all.sh`, `hpc/run_case.sh` | Slurm job-array scripts used on the Ohio Supercomputer Center (Cardinal). |
| `examples/commands.md` | The exact commands used to reconstruct each section. |

## Requirements

- Python 3.11 or newer (developed with 3.13)
- `pip install -r requirements.txt`
- A licensed Ansys Mechanical APDL installation (developed with 2026 R1) for solving.
  Geometry extraction, `--dry-run`, `--replot` and `--rescore` do not need Ansys.

## Workflow

1. **Reconstruct a section**

   ```
   python scripts/extract_kidney_outlines.py figure.pdf --panel UUO_POD3 --cut-col 1120 \
       --round-um 5 --outer-round-um 40 --out outlines
   ```

   Pelvic spaces are reconstructed as open cavities (automatic segmentation), cavities
   open at the hilum (`--seal`), collapsed spaces (`--slot-pts`, slot width `--slot-width`)
   or partly open spaces (`--hybrid`). Manual inputs can be clicked (`--pick-seed`,
   `--pick-seal`, `--pick-slot`, `--edit`) and are saved for exact reproduction.

   For ultrasound frames:

   ```
   python scripts/extract_us_outlines.py --pdf us.pdf --list --out us
   python scripts/extract_us_outlines.py --pdf us.pdf --xref 11 --name US_UUO_POD3 \
       --fov-width-mm 14.08 --pick-seed --pick-kidney --out us
   ```

2. **Check the geometry** (no Ansys needed)

   ```
   python scripts/batch_fea.py outlines/UUO_POD3 --dry-run --out check
   ```

   Reports the minimum tissue thickness between the pelvic space and the outer surface and
   saves a geometry-check figure with the landmarks and the papilla / pelvic-wall split.

3. **Solve**

   ```
   python scripts/batch_fea.py outlines/UUO_POD3 --pressure 0.00005 --out fea
   ```

   On a Slurm cluster, put each `<name>.dxf` and `<name>.csv` in `cases/` and run
   `bash hpc/submit_all.sh fea --pressure 0.00005` (options are listed in the header of
   `submit_all.sh`; edit the account and Ansys paths in `run_case.sh`).

4. **Post-process**

   ```
   python scripts/batch_fea.py fea/*/*_fields.npz --replot \
       --clim "seqv_over_p=0,3;e1=0,0.04;hydro_over_p=-1.5,1.5" --fmt png,pdf
   python scripts/batch_fea.py fea/UUO_POD3 --rescore --out fea
   ```

## Model summary

Plane strain; PLANE183 elements with a mixed u-P formulation; nearly incompressible
neo-Hookean material (E = 14 kPa, nu = 0.49 by default); uniform follower pressure on the
pelvic boundary; rigid-body constraints only; large deformation; units mm-N-MPa.

## Data availability

The histological and ultrasound images belong to the collaborating laboratory and are
[not included / available at ...]. [Derived geometries (DXF/CSV) are provided in ...]

## Citation

Please cite the article above (see `CITATION.cff`).

## License

MIT (see `LICENSE`).
