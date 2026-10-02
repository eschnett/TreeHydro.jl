# Running over MPI (added 2026-10-02, on TreeAMR's M7).
#
# TreeAMR distributes a forest's *blocks* over the ranks of a communicator
# and keeps the forest itself on every rank, and everything it does across
# ranks — the ghost exchange, the interface restriction, the regrid, the
# checkpoints, `mesh_mapreduce` — it does itself. What it cannot do is make
# a number *this package* combines from per-block values the same on every
# rank, nor agree on a decision taken from a rank's own clock. Those are the
# two things this file is for, and nothing in it is mesh machinery: it is the
# application's half of "What an application must make global itself" in
# TreeAMR's `CODE.md`.
#
# The verbs are TreeAMR's — `commrank`, `commsize`, `allgather`,
# `allgatherv` — reached through `forest.comm`, which is a
# `TreeAMR.Communicator` and a `SerialCommunicator` on a forest built
# without one. They are unexported upstream, as `launch_by_owner!` is, so
# `test/prerequisite_tests.jl` checks them by name. This package never names
# MPI: a serial run takes every function here through its one-rank branch,
# which returns its argument untouched and gathers nothing, so every serial
# number in `CODE.md` is where it was. See "Running distributed" in `CODE.md`.

# The communicator of whatever a reduction is over.
rank_comm(comm::TreeAMR.Communicator) = comm
rank_comm(forest::Forest) = forest.comm
rank_comm(fs::FieldSet) = fs.forest.comm

"""
    rank_reduce(op, over, value; present = true)

`value` combined over every rank of `over` — a forest, a field set or a
`TreeAMR.Communicator` — with `op`, **in rank order**, the same on every
rank: collective. `present = false` says this rank has nothing to
contribute (it holds no blocks), and its `value` is then left out rather
than folded in, which is TreeAMR's rule for `mesh_mapreduce` and the reason
an `init` need only be idempotent under `op`; where no rank is present the
result is `value`, which every rank then passes the same.

This is the cross-rank step of every number this package combines on the
host from per-block values — the ghost floor count, the tracking measure,
the shock radius, McNally's two diagnostics — and it is written as
TreeAMR's `combine_blocks` is: one `allgather` of a `(present, value)`
pair, folded in rank order, so the association is the package's and every
rank gets the same bits. On one rank it returns `value` itself and gathers
nothing, which is what keeps a serial run bit for bit what it was. `value`
must be `isbits`.
"""
function rank_reduce(op, over, value; present::Bool=true)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return value
    parts = TreeAMR.allgather(comm, (present, value))::Vector{Tuple{Bool,typeof(value)}}
    acc, found = value, false
    for (has, v) in parts
        has || continue
        acc = found ? op(acc, v) : v
        found = true
    end
    return acc
end

# A tuple of decisions, `true` on every rank where it is `true` on any:
# how a checkpoint taken by the wall clock is agreed, since a rank that
# entered the collective `save_checkpoint` while another went on to its
# regrid would wait forever.
agree_any(over, flags::NTuple{N,Bool}) where {N} =
    rank_reduce((a, b) -> map(|, a, b), over, flags)

# Whether this rank is the one that does what only one rank may — remove a
# file, print a line.
isroot(over) = TreeAMR.commrank(rank_comm(over)) == 0

"""
    agree_refusal(f, comm)

Run the checks `f()` on every rank, and if one of them throws an
`ArgumentError` on any rank, throw on **every** rank: the rank that refused
its own error, and the others one naming the ranks that refused.

For the checks that can come out differently on different ranks — whether a
directory or a file exists, which is a question about each rank's view of
the file system, and a node-local path is the ordinary way for two ranks to
disagree. A refusal on one rank alone would leave the others waiting in
their first collective; TreeAMR agrees its own refusals the same way. Any
other exception is a bug and is rethrown at once. Serially this is `f()`.
"""
function agree_refusal(f, comm::TreeAMR.Communicator)
    if TreeAMR.commsize(comm) == 1
        f()
        return nothing
    end
    err = nothing
    try
        f()
    catch e
        e isa ArgumentError || rethrow()
        err = e
    end
    refused = TreeAMR.allgather(comm, err !== nothing)::Vector{Bool}
    err === nothing || throw(err)
    any(refused) && throw(ArgumentError(
        "rank(s) $(join(findall(refused) .- 1, ", ")) refused the arguments, and " *
        "this rank did not: a check that reads the file system can come out " *
        "differently on different nodes, and the run stops on every rank rather " *
        "than leave the others waiting for the one that refused. Its error is " *
        "printed by that rank."))
    return nothing
end

# `v` on every rank summed element by element over the ranks, in rank
# order: for an array each rank fills only where its own blocks are and
# leaves zero elsewhere, which is what `reduce_to_grid` makes. Serially `v`
# itself.
function rank_sum!(v::Vector, over)
    comm = rank_comm(over)
    n = TreeAMR.commsize(comm)
    n == 1 && return v
    gathered = TreeAMR.allgatherv(comm, v)
    m = length(v)
    length(gathered) == n * m || throw(DimensionMismatch(
        "rank_sum! needs the same length on every rank, got $(length(gathered)) " *
        "elements from $n ranks of $m here."))
    v .= view(gathered, 1:m)
    for r in 2:n
        v .+= view(gathered, ((r - 1) * m + 1):(r * m))
    end
    return v
end
