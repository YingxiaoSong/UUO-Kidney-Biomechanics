#!/bin/bash
#SBATCH --job-name=uto_case
#SBATCH --account=PMUS0007
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/%x_%A_%a.out
#
# One array task = one section. Submitted by submit_all.sh; do not sbatch directly.

module load python/3.12
module load ansys/2026R1
source ~/uto_env/bin/activate
cd ~/uto

MAPDL=/apps/ansys/2026R1/v261/ansys/bin/ansys261
OUT_ROOT=${OUT_ROOT:-fea_all}
FEA_OPTS=$(cat "$OUT_ROOT/fea_opts.txt" 2>/dev/null)

if [ -f "$OUT_ROOT/case_list.txt" ]; then
    mapfile -t CASES < "$OUT_ROOT/case_list.txt"
else
    mapfile -t CASES < <(ls cases/*.dxf | sed 's/\.dxf$//' | sort)
fi
CASE=${CASES[$SLURM_ARRAY_TASK_ID]}
NAME=$(basename "$CASE")
OUT="$OUT_ROOT/$NAME"
mkdir -p "$OUT"

echo "case $NAME  (task $SLURM_ARRAY_TASK_ID)  options: $FEA_OPTS"
python -u batch_fea.py "$CASE" --nproc 4 --out "$OUT" \
    --exec-file "$MAPDL" --start-timeout 300 --port $((20000 + SLURM_JOB_ID % 30000)) $FEA_OPTS
