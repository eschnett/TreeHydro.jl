#!/bin/bash
# The showcase on one H200: the simulation on the device, then the movie
# rendered on the node's host cores from the frame files it wrote.
#
#     sbatch showcase/symmetry_showcase.sh                         # production, ≤ 24 h
#     SHOWCASE_CONFIG=showcase/configs/pilot.toml \
#         sbatch --partition=h200debugq --time=1:00:00 showcase/symmetry_showcase.sh
#     SHOWCASE_OUT=/mnt/beegfs/…/treehydro-showcase-<job> \
#         SHOWCASE_RESTART=latest sbatch showcase/symmetry_showcase.sh
#
# `SHOWCASE_WALLTIME=0.8` stops with a checkpoint before 0.8 h, and
# `SHOWCASE_STOP_AFTER=K` after frame K (a stage whose frames choose a target). Use the debug
# queue once, to get started, and never as a chain: a run continuing from a
# checkpoint goes to `h200q`. `h200preq` runs on the debug nodes when they are
# idle and requeues a preempted job, which with `SHOWCASE_RESTART=latest`
# continues from its newest checkpoint:
#
#     SHOWCASE_CONFIG=… SHOWCASE_OUT=… SHOWCASE_RESTART=latest \
#         sbatch --partition=h200preq --requeue showcase/symmetry_showcase.sh
#
# Submit from a checkout of its own (rsync the tree to a fresh directory,
# never into one whose jobs are running). The environment is a scratch one
# with CUDA added, as for `bin/symmetry_gpu.sh`: neither the package nor the
# showcase's own environment may gain a CUDA dependency. `SHOWCASE_RENDER=0`
# skips the render; `SHOWCASE_SIM=0` skips the simulation and renders what is
# in `SHOWCASE_OUT`.

#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Not the default 21 GB per CPU, which the h200 QOS's group memory limit
# holds pending; the host side needs a few GB.
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --job-name=treehydro-showcase
# The job log on scratch, like everything a job writes: the home directory
# has a quota that a job log filling it would break for every other job.
#SBATCH --output=/mnt/beegfs/eschnetter/claude/treehydro-showcase-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${TREEHYDRO_REPO:-$PWD}"
CONFIG="${SHOWCASE_CONFIG:-$REPO/showcase/configs/production.toml}"
case "$CONFIG" in /*) ;; *) CONFIG="$REPO/$CONFIG" ;; esac
ENVDIR="${SHOWCASE_ENV:-/mnt/beegfs/eschnetter/claude/treehydro-showcase}"
OUTDIR="${SHOWCASE_OUT:-/mnt/beegfs/eschnetter/claude/treehydro-showcase-${SLURM_JOB_ID:-local}}"
RESTART="${SHOWCASE_RESTART:-}"
# `latest` is the newest checkpoint in the output directory, or a fresh start
# when there is none yet: what makes a job on the pre-emptible `h200preq`
# safe, since a preempted job is requeued and runs this script again.
if [ "$RESTART" = latest ]; then
    RESTART=$(ls -t "$OUTDIR"/checkpoints/*.h5 2>/dev/null | head -n 1 || true)
fi
mkdir -p "$ENVDIR" "$OUTDIR"

echo "# $(hostname) $(date -Iseconds) config $CONFIG out $OUTDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
julia --version
julia --project="$ENVDIR" -e "
    using Pkg
    # The depot's registry can predate the TreeAMR release this needs.
    Pkg.Registry.update()
    Pkg.develop(path = \"$REPO\")
    for p in (\"CUDA\", \"KernelAbstractions\", \"TreeAMR\", \"HDF5\", \"CairoMakie\")
        p in keys(Pkg.project().dependencies) || Pkg.add(p)
    end
    Pkg.instantiate()
    Pkg.precompile()
    using CUDA; CUDA.versioninfo()"

JL=(julia --project="$ENVDIR" -t 16)

if [ "${SHOWCASE_SIM:-1}" != 0 ]; then
    # The device's memory once a minute beside the frame log, which records
    # the host's own peak.
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 60 \
        > "$OUTDIR/gpu.csv" &
    SMI=$!
    "${JL[@]}" "$REPO/showcase/simulate.jl" --config="$CONFIG" --out="$OUTDIR" \
        --backend=cuda ${RESTART:+--restart="$RESTART"} \
        ${SHOWCASE_WALLTIME:+--walltime-hours="$SHOWCASE_WALLTIME"} \
        ${SHOWCASE_STOP_AFTER:+--stop-after="$SHOWCASE_STOP_AFTER"}
    kill $SMI || true
fi

if [ "${SHOWCASE_RENDER:-1}" != 0 ]; then
    "${JL[@]}" "$REPO/showcase/render.jl" --frames="$OUTDIR" --out="$OUTDIR/kh_zoom.mp4"
fi
echo "# done $(date -Iseconds)"
ls -la "$OUTDIR"
