#!/bin/bash
# Submit every section in ~/uto/cases (each <name>.dxf with its <name>.csv) as one job
# array, then merge all results into <OUT_ROOT>/summary_all.csv when they are done.
#
# Usage (run on the login node, not with sbatch):
#   bash submit_all.sh                                  # baseline -> fea_all/
#   bash submit_all.sh fea_mesh05 --mesh-scale 0.5      # mesh study
#   bash submit_all.sh fea_E10 --E 0.010                # sensitivity: E = 10 kPa
#   bash submit_all.sh fea_all --clim seqv_over_p=0,8   # shared colour limits
#   MAX_PAR=4 bash submit_all.sh                        # run up to 4 sections at once (default 2)
#   CASES="UUO_POD2 CL_SHAM" bash submit_all.sh fea_redo --pressure 0.00005   # only these sections
#
# The first argument is the output folder; everything after it is passed to batch_fea.py.

cd ~/uto || exit 1
OUT_ROOT=${1:-fea_all}
shift
FEA_OPTS="$*"

shopt -s nullglob
if [ -n "$CASES" ]; then             # only the named sections, e.g. CASES="UUO_POD2 CL_SHAM"
    DXF=()
    for c in $CASES; do
        [ -f "cases/$c.dxf" ] && DXF+=("cases/$c.dxf") || echo "not found: cases/$c.dxf"
    done
else
    DXF=(cases/*.dxf)
fi
N=${#DXF[@]}
if [ "$N" -eq 0 ]; then
    echo "No DXF files in ~/uto/cases"; exit 1
fi
MISSING=0
for f in "${DXF[@]}"; do
    [ -f "${f%.dxf}.csv" ] || { echo "missing ${f%.dxf}.csv"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "Add the missing CSV files first."; exit 1; }

mkdir -p logs "$OUT_ROOT"
echo "$FEA_OPTS" > "$OUT_ROOT/fea_opts.txt"   # read by every task (avoids sbatch --export comma issues)
printf '%s\n' "${DXF[@]%.dxf}" > "$OUT_ROOT/case_list.txt"   # the sections this run computes

JID=$(sbatch --parsable --array=0-$((N-1))%${MAX_PAR:-2} --export=ALL,OUT_ROOT="$OUT_ROOT" run_case.sh)
echo "Submitted $N sections as job array $JID  (output: ~/uto/$OUT_ROOT, logs: ~/uto/logs)"

# after all tasks finish (even if some fail): one merged summary table
MID=$(sbatch --parsable --account=PMUS0007 --time=00:05:00 --ntasks=1 \
      --dependency=afterany:"$JID" --output=logs/merge_%j.out \
      --wrap="cd ~/uto && awk 'FNR==1 && NR!=1 {next} {print}' $OUT_ROOT/*/summary.csv > $OUT_ROOT/summary_all.csv && echo merged \$(( \$(wc -l < $OUT_ROOT/summary_all.csv) - 1 )) rows")
echo "Merge job $MID will write ~/uto/$OUT_ROOT/summary_all.csv"
echo "Check progress with:  squeue -u \$USER"
