# Checkpoint and restart for `evolve!` (added 2026-09-29, on TreeAMR 0.1.4's
# M9a).
#
# Everything about the *file* is upstream: TreeIOHDF5's `save_checkpoint` writes
# the forest, the evolved field sets and an application's plain data, and
# writes them atomically, durably, with element types as limbs where they
# are not HDF5 natives and with the provenance of the writer; its
# `load_checkpoint` rebuilds a forest and a field set through their own
# validating constructors. Both live in TreeIOHDF5, TreeAMR's companion
# checkpoint package (TreeAMR's own HDF5 extension until TreeAMR 0.2), a
# dependency of this package; this package never touches an HDF5 type
# itself. That is the "no mesh machinery" rule applied to I/O.
#
# What is left for this file is what only the application knows:
#
#   * **when** to write — at a chunk boundary, where a fixed-step integrator
#     holds nothing but `(t, u)`, and *before* the regrid (amended
#     2026-10-01, after TreeGeneralizedHarmonic): a restart regrids first,
#     with the criterion it is given, through the loop's own code, and so
#     begins the next chunk with exactly what the uninterrupted run began it
#     with when the criterion is unchanged;
#   * **its own run state** — the accumulators behind `evolve!`'s return
#     value, which is what makes a restarted run's *answer* the same and
#     not only its state;
#   * **the recipe** — every parameter that decides the numbers, so that a
#     restart with a different one is refused by name rather than run into
#     a different experiment that looks like the old one — and beside it
#     **the criterion**, the regridding parameters a restart may change;
#   * **the names** of the files, and their rotation.
#
# See "Checkpoint and restart" in `CODE.md`.

# The application's group in the file and the format version of what this
# package stores there. TreeAMR stores the version and never reads it; the
# check is `load_run`'s. The name mirrors TreeAMR's own group, "TreeAMR.jl".
#
# Version 2 since 2026-10-01: the checkpoint moved from after the regrid to
# before it, and the criterion left the recipe. A version-1 file holds a
# state that has already been regridded, which a restart of this version
# would regrid a second time — a different run, with the same layout — so it
# is refused rather than read.
const CHECKPOINT_APPLICATION = "TreeHydro.jl"
const CHECKPOINT_VERSION = 2

# --- names and rotation ------------------------------------------------------

# The file for the checkpoint taken after `iteration` time steps since
# `t = 0`. The step count is monotonic across restarts, which a chunk index
# would be too, but it is also what a reader of a directory listing wants to
# see — how far the run got — and it is the Cactus convention. Ten digits
# sort lexically as well as numerically up to 10¹⁰ steps; the pattern below
# reads any number of them.
checkpoint_filename(prefix, iteration::Integer) =
    "$prefix.it$(lpad(iteration, 10, '0')).h5"

# `s` quoted for a regular expression, so that a prefix with a `.` or a `+`
# in it matches itself and nothing else.
regex_quote(s::AbstractString) = replace(s, r"[\\^$.|?*+()\[\]{}]" => s"\\\0")

"""
    checkpoint_files(prefix) -> Vector{Tuple{Int,String}}

The checkpoint files of `prefix`, as `(iteration, path)` pairs sorted by
iteration: the files in `prefix`'s directory whose names are exactly
`"<basename>.it<digits>.h5"`. Nothing else matches — in particular not
TreeIOHDF5's `"….h5.partial"`, the file a write in progress or a failed one
leaves, not the part files `"….h5.<save id>.<j>.h5"` it writes beside a
distributed checkpoint (whose index is the file listed), and not another
prefix that merely starts with this one. A directory that does not exist
holds no checkpoints.

It reads the directory on the rank that calls it, so over MPI the
checkpoint directory is one every rank sees the same, as TreeIOHDF5's own
checkpoints need it to be.
"""
function checkpoint_files(prefix::AbstractString)
    dir, base = splitdir(prefix)
    found = Tuple{Int,String}[]
    isempty(base) && return found
    listed = isempty(dir) ? "." : dir
    isdir(listed) || return found
    pattern = Regex("^" * regex_quote(base) * raw"\.it(\d+)\.h5$")
    for name in readdir(listed)
        m = match(pattern, name)
        m === nothing && continue
        iteration = tryparse(Int, m.captures[1])
        iteration === nothing && continue
        path = isempty(dir) ? name : joinpath(dir, name)
        isfile(path) && push!(found, (iteration, path))
    end
    return sort!(found)
end

"""
    latest_checkpoint(prefix)

The checkpoint file of `prefix` with the highest iteration, or `nothing` if
there is none — which is what makes a job chain one command for every job,
the first included:

```julia
using TreeHydro
r = evolve!(case; …, checkpoint_path_prefix = prefix,
            max_walltime_seconds = 23.5 * 3600,
            restart_file = latest_checkpoint(prefix))
r.finished || exit(3)          # resubmit
```

The first job finds nothing and starts from the initial data; every later
one continues from where the previous one stopped. See "Checkpoint and
restart" in `CODE.md`.
"""
function latest_checkpoint(prefix::AbstractString)
    files = checkpoint_files(prefix)
    return isempty(files) ? nothing : last(files)[2]
end

# Delete every checkpoint file of `prefix` but `keep` and the newest
# `num_keep − 1` others, and return the paths deleted. Run only after a write
# has succeeded, so that the newest complete checkpoint is never among them.
# The file just written survives even when it is not the newest — a run
# restarted from an older checkpoint while newer ones exist — because it is
# the only one this run can vouch for. It includes files an earlier job left
# behind, which is the point in a job chain: the disk holds `num_keep` files
# whichever job wrote them.
#
# The file just written is recognised by its *name*, not its path: every
# file listed is in `prefix`'s directory by construction, and a path string
# is not a file's identity — `run//sedov` lists as `run/sedov`, and with
# `num_keep = 1` a comparison of paths deleted the checkpoint just written.
#
# A distributed checkpoint is an index and its part files (TreeAMR's M7):
# `"<index>.<save id>.<j>.h5"` beside the index, one per I/O group, the save
# id 32 lowercase hex digits. Removing an index removes its parts with it,
# and any orphan of that index's name — a part left by a save that failed
# before its index was renamed into place — since nothing will ever read
# them once the index is gone. A serial checkpoint has no part files. Over
# MPI rank 0 alone calls this (amended 2026-10-02).
function rotate_checkpoints!(prefix::AbstractString, num_keep::Integer;
                             keep::AbstractString)
    others = [path for (_, path) in checkpoint_files(prefix)
              if basename(path) != basename(keep)]
    removed = others[1:max(0, length(others) - (num_keep - 1))]
    for path in removed
        dir, name = splitdir(path)
        listed = isempty(dir) ? "." : dir
        part = Regex("^" * regex_quote(name) * raw"\.[0-9a-f]{32}\.[0-9]+\.h5$")
        rm(path; force=true)
        for other in readdir(listed)
            occursin(part, other) || continue
            rm(isempty(dir) ? other : joinpath(dir, other); force=true)
        end
    end
    return removed
end

# --- exact reals as plain data -------------------------------------------------

const NativeFloat = Union{Float16,Float32,Float64}

# The one native type an `isbits` type is made of throughout, with no
# padding — `Float32` for MultiFloats' `Float32x2`, an `NTuple{2,Float32}` of
# limbs — and `nothing` for anything else. TreeIOHDF5 applies the
# same rule to a field set's element type; it is restated here because that
# function is internal to TreeIOHDF5, and its `write_plain`, which
# the run state goes through, refuses a MultiFloat scalar.
function limb_type(::Type{T}) where {T}
    T <: Union{NativeFloat,Base.BitInteger} && return T
    (isstructtype(T) && isconcretetype(T) && fieldcount(T) > 0) || return nothing
    F = nothing
    size = 0
    for i in 1:fieldcount(T)
        S = fieldtype(T, i)
        L = limb_type(S)
        (L === nothing || (F !== nothing && L !== F)) && return nothing
        F = L
        size += sizeof(S)
    end
    return size == sizeof(T) ? F : nothing
end

function limbs_of(::Type{R}) where {R}
    F = isbitstype(R) ? limb_type(R) : nothing
    F === nothing && throw(ArgumentError(
        "a checkpoint cannot store the reals of $R exactly: it is neither a " *
        "native float nor an isbits type made of one native type throughout " *
        "with no padding, which is stored as its limbs (Float32x2 as two " *
        "Float32). A file stores bits, and no other type could be read back " *
        "bit for bit."))
    return (F, sizeof(R) ÷ sizeof(F))
end

"""
    plain_reals(xs)

Values of the run's real type as plain data that read back **bit for bit**:
a vector of a native float as itself, and a vector of any other `isbits` real
made of one native float throughout (MultiFloats' `Float32x2`) as the matrix
of its limbs, `(nlimbs, n)`, limb first as in memory. A scalar is stored as a
one-element vector and a tuple as a vector. [`from_plain_reals`](@ref) is
the inverse. This is how a `Float32x2` run's accumulators reach a file whose
plain data refuse a MultiFloat scalar.
"""
plain_reals(x::Real) = plain_reals([x])
plain_reals(xs::Tuple) = plain_reals(collect(xs))
function plain_reals(xs::AbstractVector{R}) where {R}
    R <: NativeFloat && return collect(xs)
    F, n = limbs_of(R)
    return collect(reshape(reinterpret(F, collect(xs)), n, length(xs)))
end

"""
    from_plain_reals(R, a) -> Vector{R}

The inverse of [`plain_reals`](@ref): the vector of `R` that `a` stores,
refused if `a` is not what `plain_reals` makes of an `R`.
"""
function from_plain_reals(::Type{R}, a) where {R}
    if R <: NativeFloat
        a isa AbstractVector{R} || throw(ArgumentError(
            "a checkpoint value is a $(typeof(a)) where a vector of $R was " *
            "expected: the file was written by a run in another type, or damaged."))
        return collect(a)
    end
    F, n = limbs_of(R)
    (a isa AbstractMatrix{F} && size(a, 1) == n) || throw(ArgumentError(
        "a checkpoint value is a $(typeof(a)) of size $(size(a)) where the " *
        "$n $F limbs of a vector of $R were expected: the file was written by " *
        "a run in another type, or damaged."))
    return collect(reinterpret(R, vec(a)))
end

from_plain_scalar(::Type{R}, a) where {R} = only(from_plain_reals(R, a))

function from_plain_tuple(::Type{R}, a, ::Val{n}) where {R,n}
    v = from_plain_reals(R, a)
    length(v) == n || throw(ArgumentError(
        "a checkpoint tuple has $(length(v)) entries where $n were expected: " *
        "the file was written for another dimension, or damaged."))
    return ntuple(i -> v[i], Val(n))
end

# A type's name as a module importing nothing but Base prints it —
# `Float64`, `MultiFloats.MultiFloat{Float32, 2}` — and not `string(T)`,
# which qualifies a name or not according to what the writer happened to have
# imported into `Main`. TreeIOHDF5 names element types the same way,
# for the same reason.
module TypeNames end
type_name(::Type{T}) where {T} = sprint(show, T; context=:module => TypeNames)

# A parameter struct — the equation of state, the floors — as its type's
# name, its field names and its field values, each real through
# `plain_reals`. The names are a list of strings rather than the keys of a
# group, which keeps `ρ_atm` out of the file's link names.
function plain_struct(x)
    names = fieldnames(typeof(x))
    values = map(n -> plain_field(getfield(x, n)), names)
    return (; kind=String(nameof(typeof(x))), names=collect(String.(names)),
            values=Tuple(values))
end
plain_field(x::AbstractFloat) = plain_reals(x)
plain_field(x::Union{Integer,Symbol,AbstractString,Nothing}) = x
plain_field(x::Tuple) = map(plain_field, x)
plain_field(x) = throw(ArgumentError(
    "a case parameter of type $(typeof(x)) has no plain form a checkpoint can " *
    "compare: extend `plain_field` for it."))

# --- the recipe ------------------------------------------------------------------

"""
    run_recipe(T, case, D; N, G, roots, ops, chunk, cfl, limiter, riemann,
               fixup, reset, accounting)

Every parameter of an [`evolve!`](@ref) call that decides the numbers, as
plain data, **except the regridding criterion** ([`run_criterion`](@ref)),
which a restart may change: the working type by name, the mesh (`D`, `N`,
`G`, the roots, the periodicity, the reflecting faces, the extents), the
case's equation of state, floors and speed headroom, the cadence and the
step (`chunk`, `cfl`), the scheme (`limiter`, `riemann`, `fixup`, `reset`,
`accounting`) and the operators. `t_end` is not in it, because a restart may
move it; nor are `backend`, `maxpasses` or the observer, which do not change
a number of the run once its initial-data cycle is over.

Every real goes through `T` first and then [`plain_reals`](@ref), so a
`2//25` given to one call and a `T(2//25)` given to the next compare equal,
as they are the same run. The keys are ASCII (`epsilon`, `epsilon_g`) so that
a reader in another language finds them.

What the recipe **cannot** hold is the case's closures — `initial`,
`boundary` and `reference` — so a restart with the same parameters and
different initial data is not detected. The initial data do not enter a
restarted run at all, but the boundary hook does: the caller is trusted to
pass the same case.
"""
function run_recipe(::Type{T}, case::HydroCase, ::Val{D}; N, G, roots, ops, chunk,
                    cfl, limiter, riemann, fixup, reset, accounting) where {T,D}
    r(x) = plain_reals(T(x))
    tupleD(x) = x isa Integer ? ntuple(_ -> Int(x), D) : ntuple(d -> Int(x[d]), D)
    return (; float_type=type_name(T), D=Int(D), N=Int(N), G=tupleD(G),
            roots=tupleD(roots), periodic=case.periodic,
            reflecting=case.reflecting,
            extents=plain_reals([x for ext in case.extents for x in ext]),
            eos=plain_struct(case.eos), floors=plain_struct(case.floors),
            speed_headroom=r(case.speed_headroom), chunk=r(chunk), cfl=r(cfl),
            limiter=limiter, riemann=riemann, fixup=Bool(fixup), reset=reset,
            accounting=Bool(accounting),
            ops=(; family=Symbol(ops.family), prolongation=Int(ops.prolongation),
                 restriction=Int(ops.restriction)))
end

"""
    run_criterion(T; refine_tol, coarsen_tol, maxlevel_cap, ε, ε_g, buffer)

The parameters a restart **may** change (decided 2026-10-01, after
TreeGeneralizedHarmonic): the regridding criterion — the two thresholds, the
cap, the indicator's two `ε`s and the travelling margin `buffer` — as plain
data, each real through `T` and then [`plain_reals`](@ref) as in
[`run_recipe`](@ref). The checkpoint is written before the regrid, so a
restart with a changed criterion regrids with it first, before its next
step; the change is reported field by field and returned as
`criterion_changed`. The keys are the recipe's (`epsilon`, `epsilon_g`).
"""
function run_criterion(::Type{T}; refine_tol, coarsen_tol, maxlevel_cap, ε, ε_g,
                       buffer) where {T}
    r(x) = plain_reals(T(x))
    return (; refine_tol=r(refine_tol), coarsen_tol=r(coarsen_tol),
            maxlevel_cap=Int(maxlevel_cap), epsilon=r(ε), epsilon_g=r(ε_g),
            buffer=buffer === nothing ? nothing : Int(buffer))
end

# A plain value as a message prints it: a one-element vector — how a scalar
# real is stored — as its element.
describe_plain(x::AbstractVector) = length(x) == 1 ? repr(only(x)) : repr(x)
describe_plain(x) = repr(x)

# The fields of two plain named tuples that differ, each with a sentence
# naming both values: the refusal of a recipe and the report of a criterion
# are the same comparison.
function plain_differences(saved, current)
    diffs = Tuple{Symbol,String}[]
    for k in unique((keys(saved)..., keys(current)...))
        a = haskey(saved, k) ? saved[k] : missing
        b = haskey(current, k) ? current[k] : missing
        isequal(a, b) && continue
        was = a === missing ? "absent" : describe_plain(a)
        is = b === missing ? "absent" : describe_plain(b)
        push!(diffs, (k, "`$k` is $was in the checkpoint and $is in this call"))
    end
    return diffs
end

"""
    check_recipe(saved, current, path)

Refuse a restart whose parameters differ from the checkpoint's, with one
`ArgumentError` that names **every** field that differs and both of its
values — so that a job script with two wrong keywords is fixed in one round
and not two. Equality is `isequal` on the plain forms, which for reals is
equality of the bits in the run's type.
"""
function check_recipe(saved, current, path)
    diffs = plain_differences(saved, current)
    isempty(diffs) || throw(ArgumentError(
        "restart_file $(repr(path)) was written by a run with other parameters: " *
        join(last.(diffs), "; ") * ". A restart continues the saved run, and a " *
        "run continued with another parameter would be a different experiment " *
        "that looks like the old one, so it must be called with the same case " *
        "and the same keywords — only t_end and the regridding criterion " *
        "(refine_tol, coarsen_tol, maxlevel_cap, ε, ε_g, buffer) may change (and " *
        "backend, maxpasses and the observer, which decide no number of the " *
        "run). The case's closures, its initial data, boundary hook and " *
        "reference, cannot be compared and are trusted to be the same."))
    return nothing
end

"""
    check_criterion(saved, current, path) -> Vector{Symbol}

The fields of the regridding criterion that a restart changes, in the
criterion's order — empty when it changes none — each reported with both of
its values in one `@info`. A change is allowed (see [`run_criterion`](@ref)),
and said, so that a log shows where a run stopped being the one it began as.
"""
function check_criterion(saved, current, path)
    changes = plain_differences(saved, current)
    isempty(changes) || @info "restarting $(repr(path)) with a changed regridding " *
                              "criterion, which the first regrid, before the next " *
                              "step, uses: " * join(last.(changes), "; ")
    return first.(changes)
end

# --- the run state -----------------------------------------------------------------

# The accumulators behind `evolve!`'s return value at the end of chunk `c`,
# before its regrid: everything a restart needs for its answer, and not only
# its state, to be the uninterrupted run's — and, in `λ_end_history[end]`,
# the speed its first regrid derives the margin from. Every real through
# `plain_reals`.
#
# Every value must be the same on every rank, which TreeIOHDF5's plain data
# require of a distributed checkpoint and refuse otherwise: `reset_hits` is
# the total over the ranks, which the caller sums (the reset counts per
# rank; see `ResetAccounting`), and everything else is global already.
function run_state(; chunk, t, nsteps, nregrids, floor_hits, ghost_hits, passes,
                   converged, reset_hits, injection, λ_initial, tracking, drift,
                   scales, totals0, nblocks_history, buffer_history, λ_history,
                   λ_end_history)
    return (; chunk=Int(chunk), t=plain_reals(t), nsteps=Int(nsteps),
            nregrids=Int(nregrids), floor_hits=Int(floor_hits),
            ghost_hits=Int(ghost_hits), passes=Int(passes), converged=Bool(converged),
            reset_hits=Int(reset_hits), injection=plain_reals(injection),
            lambda_initial=plain_reals(λ_initial), tracking=plain_reals(tracking),
            drift=plain_reals(drift), scales=plain_reals(scales),
            totals0=plain_reals(totals0), nblocks_history=Vector{Int}(nblocks_history),
            buffer_history=Vector{Int}(buffer_history),
            lambda_history=plain_reals(λ_history),
            lambda_end_history=plain_reals(λ_end_history))
end

# --- writing and reading -------------------------------------------------------------

"""
    save_run(path, forest, U, u; recipe, criterion, run, filters = (), sync = true,
             io = :node)

One checkpoint: the forest, the conserved state `U` with its state vector
`u`, and this package's plain data `(; recipe, criterion, run)`, through
TreeIOHDF5's `save_checkpoint` — atomically, so a failed write leaves the
previous file alone. Only `U` is saved: the primitive set and the fluxes are
scratch that [`update_primitives!`](@ref) rebuilds from `u`, ghosts
included — on a restart before its first regrid, as at the start of every
chunk. `u` and not `U.work`: the state vector is the integrator's, and the
authoritative copy.

Over a distributed forest it is collective, and `io` is TreeIOHDF5's grouping
of the ranks into I/O processes, each writing a part file beside the index
at `path` (TreeAMR's M7); serially it is one file whatever `io` says.
"""
function save_run(path, forest, U, u; recipe, criterion, run, filters=(),
                  sync::Bool=true, io=:node)
    return save_checkpoint(path, forest; fieldsets=("U" => (U, u),),
                           application=CHECKPOINT_APPLICATION => CHECKPOINT_VERSION,
                           data=(; recipe=recipe, criterion=criterion, run=run),
                           filters=filters, sync=sync, io=io)
end

"""
    load_run(path, T; backend = CPU(), comm = nothing)
        -> (; forest, U, u, recipe, criterion, run)

Read a checkpoint written by [`save_run`](@ref), refusing one that is not
this package's — another application's, or a format version other than
$(CHECKPOINT_VERSION) — with the reason. The field set comes back in the type
it was saved in; whether that is `T` is the recipe's to say, so that the
refusal names it with the rest (see [`check_recipe`](@ref)). `T` is passed to
TreeIOHDF5 as the one type it may have to name, which is harmless for a native
float.

`comm` distributes the forest it reads over a communicator, as
[`evolve!`](@ref)'s keyword does: collective then, and a checkpoint written at
any rank count loads at any other, or serially (TreeAMR's M7). Every refusal
below is taken from what every rank read alike, so it is every rank's.
"""
function load_run(path::AbstractString, ::Type{T}; backend=CPU(),
                  comm=nothing) where {T}
    ck = load_checkpoint(path; backend=backend, types=(T,), comm=comm)
    name, version = ck.application
    written = "It was written by TreeAMR " *
              "$(something(ck.provenance.treeamr_version, "(unknown version)")) " *
              "on $(ck.provenance.created)."
    name == CHECKPOINT_APPLICATION || throw(ArgumentError(
        "$(repr(path)) is a checkpoint of the application $(repr(name)), not of " *
        "$(CHECKPOINT_APPLICATION): its run state is that application's, and this " *
        "package cannot continue a run it did not write. $written"))
    version == CHECKPOINT_VERSION || throw(ArgumentError(
        "$(repr(path)) stores TreeHydro's run state in format version $version, " *
        "and this version of TreeHydro reads version $(CHECKPOINT_VERSION) only. " *
        (version == 1 ?
         "Version 1 was written after the regrid at its chunk boundary and " *
         "version 2 before it, so a restart of this version would regrid a " *
         "state that has already been regridded — another run. " : "") *
        "A file from another TreeHydro is read by that version — TreeIOHDF5's " *
        "`checkpoint_environment(path, dir)` writes the environment that wrote " *
        "it. $written"))
    (ck.data isa NamedTuple && haskey(ck.data, :recipe) &&
     haskey(ck.data, :criterion) && haskey(ck.data, :run) &&
     haskey(ck.fieldsets, "U")) || throw(ArgumentError(
        "$(repr(path)) names $(CHECKPOINT_APPLICATION) version $version but holds " *
        "no recipe, no criterion, no run state or no field set \"U\": the file " *
        "is damaged, or was not written by `evolve!`. $written"))
    U = ck.fieldsets["U"].fieldset
    u = ck.fieldsets["U"].state
    # Every version-2 recipe records the reflecting faces: a file from before
    # they existed is version 1 and refused above, so the reading of a recipe
    # without them as "none" (added 2026-09-29) has nothing left to read.
    return (; forest=ck.forest, U=U, u=u, recipe=ck.data.recipe,
            criterion=ck.data.criterion, run=ck.data.run)
end

# --- the keywords ------------------------------------------------------------------------

# The checkpoint keywords of `evolve!`, refused up front — before the
# initial-data cycle — so that a job script's mistake costs a second and not
# the queue wait and the hours before the first write.
function check_checkpoint_keywords(; checkpoint_path_prefix, checkpoint_every_chunks,
                                   checkpoint_interval_seconds, max_walltime_seconds,
                                   num_checkpoints_keep, restart_file)
    prefix = checkpoint_path_prefix
    triggers = (checkpoint_every_chunks, checkpoint_interval_seconds,
                max_walltime_seconds)
    if prefix === nothing
        all(isnothing, triggers) || throw(ArgumentError(
            "checkpoint_every_chunks, checkpoint_interval_seconds or " *
            "max_walltime_seconds was given without checkpoint_path_prefix: the " *
            "run would be asked to write a checkpoint, or to stop and leave one, " *
            "with nowhere to write it. Pass checkpoint_path_prefix, such as " *
            "\"run/sedov\" for files run/sedov.it0000001234.h5."))
    else
        prefix isa AbstractString || throw(ArgumentError(
            "checkpoint_path_prefix must be a string, got $(repr(prefix))."))
        dir, base = splitdir(prefix)
        isempty(base) && throw(ArgumentError(
            "checkpoint_path_prefix $(repr(prefix)) ends in a directory separator: " *
            "it is the start of each file's name, not a directory, so it needs a " *
            "stem — \"run/sedov\" writes run/sedov.it0000001234.h5."))
        isempty(dir) || isdir(dir) || throw(ArgumentError(
            "the directory of checkpoint_path_prefix, $(repr(dir)), does not " *
            "exist: the first checkpoint would fail to be written hours into the " *
            "run. Create it first."))
        any(!isnothing, triggers) || throw(ArgumentError(
            "checkpoint_path_prefix was given but no checkpoint_every_chunks, " *
            "checkpoint_interval_seconds or max_walltime_seconds: nothing would " *
            "ever write a checkpoint. Pass at least one of them."))
    end
    checkpoint_every_chunks === nothing ||
        (checkpoint_every_chunks isa Integer && checkpoint_every_chunks ≥ 1) ||
        throw(ArgumentError(
            "checkpoint_every_chunks must be an integer of at least 1, got " *
            "$(repr(checkpoint_every_chunks)): it is how many chunks lie between " *
            "two checkpoints."))
    checkpoint_interval_seconds === nothing ||
        (checkpoint_interval_seconds isa Real && checkpoint_interval_seconds ≥ 0) ||
        throw(ArgumentError(
            "checkpoint_interval_seconds must be a non-negative number, got " *
            "$(repr(checkpoint_interval_seconds)): it is the wall-clock time " *
            "between two checkpoints, and 0 writes one at every chunk boundary."))
    max_walltime_seconds === nothing ||
        (max_walltime_seconds isa Real && max_walltime_seconds > 0) ||
        throw(ArgumentError(
            "max_walltime_seconds must be a positive number, got " *
            "$(repr(max_walltime_seconds)): it is the job's wall-time limit, " *
            "from the call to evolve!, before which the run writes a checkpoint " *
            "and stops."))
    (num_checkpoints_keep isa Integer && num_checkpoints_keep ≥ 1) ||
        throw(ArgumentError(
            "num_checkpoints_keep must be an integer of at least 1, got " *
            "$(repr(num_checkpoints_keep)): the newest checkpoint is the one a " *
            "restart needs, so it is always kept."))
    if restart_file !== nothing
        restart_file isa AbstractString || throw(ArgumentError(
            "restart_file must be a path or nothing, got $(repr(restart_file))."))
        isfile(restart_file) || throw(ArgumentError(
            "restart_file $(repr(restart_file)) does not exist. To start from the " *
            "initial data when there is no checkpoint yet, pass " *
            "`restart_file = latest_checkpoint(prefix)`, which is nothing then."))
    end
    return nothing
end
