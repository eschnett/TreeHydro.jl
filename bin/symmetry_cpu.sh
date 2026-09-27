#!/bin/bash
# What one Symmetry AMD node can do with TreeHydro (step 14): a block-size
# scan of the entropy wave in three dimensions, then a thread scan at the
# best block size, then the same step on a two-level mesh and the Sedov
# blast. One 64-core EPYC 7543 node, 8 NUMA domains.
#
#     sbatch bin/symmetry_cpu.sh
#     sbatch --partition=amdq --time=4:00:00 bin/symmetry_cpu.sh   # longer
#
# Run from a checkout of its own (`TREEHYDRO_REPO`, default `$PWD`): a
# directory whose sources are rsynced while its jobs precompile has its
# package image rewritten under them. The environment is one of its own
# under scratch that `Pkg.develop`s the checkout, so the package's own
# environment is never touched.
#
# Threads are pinned by Julia (`JULIA_EXCLUSIVE=1` with
# `srun --cpu-bind=none`), and TreeAMR's block ownership then places each
# block's pages by first touch on the core that computes it — the fastest
# of the placements TreeAMR measured. Every row is
# `bin/benchmark.jl`'s tab-separated format; the outputs are echoed at the
# end of the job log and kept in `$OUTDIR`.

#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treehydro-cpu
#SBATCH --output=treehydro-cpu-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEHYDRO_REPO:-$PWD}"
ENVDIR="${TREEHYDRO_CPU_ENV:-/mnt/beegfs/eschnetter/claude/treehydro-cpu}"
OUTDIR="${TREEHYDRO_OUT:-/mnt/beegfs/eschnetter/claude/treehydro-bench-${SLURM_JOB_ID:-local}}"
mkdir -p "$ENVDIR" "$OUTDIR"

echo "# $(hostname) $(date -Iseconds)"
lscpu | grep -E 'Model name|^Socket|^NUMA node\(s\)|Thread\(s\) per core'
julia --version
# `KernelAbstractions` as a direct dependency, because `bin/backend.jl`
# loads it by name and a package reachable only through TreeHydro cannot
# be `using`ed from a script.
julia --project="$ENVDIR" -e "
    using Pkg
    Pkg.develop(path = \"$REPO\")
    \"KernelAbstractions\" in keys(Pkg.project().dependencies) ||
        Pkg.add(\"KernelAbstractions\")
    Pkg.instantiate()
    Pkg.precompile()"

export JULIA_EXCLUSIVE=1
export OPENBLAS_NUM_THREADS=1
bench() {  # bench <threads> <outfile> <args...>
    local t=$1 out=$2; shift 2
    srun --cpu-bind=none julia -t "$t" --project="$ENVDIR" \
        "$REPO/bin/benchmark.jl" "$@" > "$OUTDIR/$out"
}

# The block-size scan: `N³` cells per block, the block count from `roots³`,
# from about 2¹⁵ to 2²⁷ cells, in one process so that compilation is paid
# once; rows are flushed per point, so a point that fails keeps the rows
# before it.
SCAN="${TREEHYDRO_SCAN:-8:8,8:16,8:32,12:8,12:16,12:24,16:4,16:8,16:16,16:32,24:4,24:8,24:16,32:2,32:4,32:8,32:16,48:2,48:4,48:8,64:2,64:4,64:8}"
bench 64 scan.tsv --dim=3 --case=wave --scan="$SCAN" --reps=3 --steps=2

# The thread scan at a representative size, and at the size the scan finds
# best (`TREEHYDRO_BEST`, `N:roots`).
BEST="${TREEHYDRO_BEST:-32:8}"
for t in 1 2 4 8 16 32 64; do
    bench "$t" "threads-$t.tsv" --dim=3 --case=wave --scan="$BEST" --reps=3 --steps=2
done

# The same size on a two-level mesh, and the blast, which adds the boundary
# hook and the floors; and the whole tracked blast through the driver.
bench 64 refined.tsv --dim=3 --case=wave --refined --scan="$BEST" --reps=3 --steps=2
bench 64 sedov.tsv --dim=3 --case=sedov --refined --scan="$BEST" --reps=3 --steps=2
bench 64 driver.tsv --dim=3 --case=sedov --scan=8:4 --reps=2 --steps=2 --driver

echo "=== results ($OUTDIR) ==="
for f in "$OUTDIR"/*.tsv; do
    echo "--- $(basename "$f")"
    cat "$f"
done
