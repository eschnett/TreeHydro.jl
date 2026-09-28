#!/bin/bash
# What one H200 can do with TreeHydro (step 14): the block-size scan of
# `bin/symmetry_cpu.sh` on the device at `Float64` and `Float32`, the
# device test file against the host, and the same scan on the node's own
# host cores for the comparison.
#
#     sbatch bin/symmetry_gpu.sh
#     sbatch --partition=h200q --time=4:00:00 bin/symmetry_gpu.sh   # longer
#
# h200debugq holds 2 nodes of 8× H200 with a 1-hour limit; one GPU is used.
# The environment is one of its own under scratch, with CUDA added: neither
# the package nor its test environment may gain a CUDA dependency, since CI
# has no GPU and would install it on every run for nothing.

#SBATCH --partition=h200debugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --job-name=treehydro-gpu
#SBATCH --output=treehydro-gpu-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEHYDRO_REPO:-$PWD}"
ENVDIR="${TREEHYDRO_GPU_ENV:-/mnt/beegfs/eschnetter/claude/treehydro-gpu}"
OUTDIR="${TREEHYDRO_OUT:-/mnt/beegfs/eschnetter/claude/treehydro-bench-${SLURM_JOB_ID:-local}}"
mkdir -p "$ENVDIR" "$OUTDIR"

echo "# $(hostname) $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv
julia --version
julia --project="$ENVDIR" -e "
    using Pkg
    Pkg.develop(path = \"$REPO\")
    for p in (\"CUDA\", \"KernelAbstractions\", \"TreeAMR\", \"MultiFloats\", \"Test\", \"Random\")
        p in keys(Pkg.project().dependencies) || Pkg.add(p)
    end
    Pkg.instantiate()
    Pkg.precompile()
    using CUDA; CUDA.versioninfo()"

JL=(julia --project="$ENVDIR")

echo "=== device tests, CUDA against the host ==="
TREEHYDRO_TEST_BACKEND=cuda "${JL[@]}" -t 8 -e "
    using Test, TreeAMR, TreeHydro
    using KernelAbstractions: @kernel, @index
    @testset \"device\" begin
        include(\"$REPO/test/device_tests.jl\")
    end"

# `TREEHYDRO_BENCH=0` stops here: the device tests alone take minutes, the
# scans below most of the hour.
if [ "${TREEHYDRO_BENCH:-1}" = 0 ]; then
    exit 0
fi

SCAN="${TREEHYDRO_SCAN:-8:8,8:16,8:32,12:8,12:16,12:24,16:4,16:8,16:16,16:32,24:4,24:8,24:16,32:2,32:4,32:8,32:16,48:2,48:4,48:8,64:2,64:4}"
for T in f64 f32; do
    "${JL[@]}" "$REPO/bin/benchmark.jl" --backend=cuda --type=$T --dim=3 \
        --case=wave --scan="$SCAN" --reps=3 --steps=2 > "$OUTDIR/cuda-$T.tsv"
done
BEST="${TREEHYDRO_BEST:-32:8}"
"${JL[@]}" "$REPO/bin/benchmark.jl" --backend=cuda --type=f64 --dim=3 --case=sedov \
    --refined --scan="$BEST" --reps=3 --steps=2 > "$OUTDIR/cuda-sedov.tsv"

echo "=== the same on the node's 16 host cores ==="
JULIA_EXCLUSIVE=1 "${JL[@]}" -t 16 "$REPO/bin/benchmark.jl" --dim=3 --case=wave \
    --scan="$BEST" --reps=3 --steps=2 > "$OUTDIR/host16.tsv"

echo "=== results ($OUTDIR) ==="
for f in "$OUTDIR"/*.tsv; do
    echo "--- $(basename "$f")"
    cat "$f"
done
