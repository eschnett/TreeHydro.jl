# What the resolved TreeAMR has to provide before any of the scheme can be
# written.
#
# `Project.toml` takes TreeAMR from the General registry at `0.1.1`, so
# what these tests run against is a released version and *not* the checkout
# at `~/src/jl/TreeAMR` — nor, since that release, whatever is on its
# `main`. Two things landed upstream as this package's prerequisites (see
# "Upstream prerequisites" in `CODE.md`), and a release that lost either of
# them would otherwise be found by a `MethodError` in the middle of step 1
# rather than here. The rest of the
# M8 surface the scheme is written against is checked as a list of names,
# which is the cheapest thing that fails when a signature is renamed
# upstream.
#
# Nothing here is hydrodynamics. These are claims about the mesh, and about
# the two host-side helpers this package keeps of its own.

using KernelAbstractions: @kernel, @index

const PREREQ_DIMS = (1, 2)

# A forest small enough to check point by point and refined enough that the
# blocks do not all have the same spacing. Two roots per dimension, one of
# them refined, 2:1 balanced — the shape `hydro_forest` will have.
function prereq_forest(::Val{D}; N=8) where {D}
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, [first(forest.leaves)])
    balance!(forest)
    return forest
end

# The layouts this package actually uses, one of each kind: the conserved
# state (cell-centered, `G = 2`) and a flux set (face-centered, `G = 0`
# along the stagger and 1 across it). The second is the interesting one —
# a per-dimension `G` with a zero in it, which a uniform `G` papers over.
prereq_layouts(::Val{1}) = ((cellcentered(1), (2,)), (facecentered(1, 1), (0,)))
prereq_layouts(::Val{2}) = ((cellcentered(2), (2, 2)), (facecentered(2, 1), (0, 1)))

# A state that is only definable as a whole — the shape of `prim2con`, and
# the reason `AllVariables` exists. `D + 2` values, no two of them equal,
# so a form that filled the wrong slot would show.
prereq_state(x::NTuple{D}) where {D} =
    (sum(x) + 1, ntuple(d -> d * x[d] - 1, Val(D))..., prod(x) + 2)

# The tag a stored-range kernel writes: a number naming the point's own
# index and block, never zero. A point the launch never reached keeps the
# zero `FieldSet` allocated it; a point reached with an index shifted by
# `G` — the mistake the default and `closed = true` forms would make here —
# gets some other point's tag.
prereq_tag(idx::NTuple{D,<:Integer}, b::Integer) where {D} =
    1000000 * b + sum(ntuple(d -> idx[d] * 100^(d - 1), Val(D)))

@kernel function prereq_stored_kernel!(work, ::Val{D}) where {D}
    I = @index(Global, NTuple)                 # already a stored index
    b = I[D + 1]
    idx = ntuple(d -> I[d], Val(D))
    work[idx..., 1, b] = prereq_tag(idx, b)
end

@testset "Coordinate callbacks run once per point for all variables: D=$D" for D in
                                                                               PREREQ_DIMS
    # Guards the loss of `AllVariables`, and the subtler failure of a form
    # that fills a field set with *different numbers* from the per-variable
    # one: the initial data of every case goes through
    # `AllVariables(x -> prim2con(eos, initial(x)))`, so the two forms
    # agreeing only to roundoff would put a floor under every error this
    # package measures.
    forest = prereq_forest(Val(D))
    for (centering, G) in prereq_layouts(Val(D))
        per = FieldSet{Float64}(forest, D + 2; G=G, centering=centering)
        whole = FieldSet{Float64}(forest, D + 2; G=G, centering=centering)
        fill_by_coordinates!((x, v) -> prereq_state(x)[v], per)
        fill_by_coordinates!(AllVariables(prereq_state), whole)
        @test isequal(whole.work, per.work)
        @test any(!iszero, whole.work)         # and it filled something
    end

    # The length check is the one thing the wrapper can say that the bare
    # callback cannot, and it says it on the host, before the launch.
    fs = FieldSet{Float64}(forest, D + 2; G=2, centering=cellcentered(D))
    @test_throws "one value per variable" fill_by_coordinates!(
        AllVariables(x -> (sum(x),)), fs)
end

@testset "A launch reaches every stored point, ghosts included: D=$D" for D in PREREQ_DIMS
    # Guards the loss of `map_blocks!(…; stored = true)`, and the two ways
    # it could be there and wrong: a launch covering only the owned range
    # would leave the ghosts at zero, and one whose index still needed `G`
    # added would write every tag at the wrong point — in bounds, and
    # silently. The `con2prim` pass of step (2) of the right-hand side is
    # the consumer: the reconstruction reads primitives two cells into the
    # neighbours, so the recovery has to have run there.
    N = 8
    forest = prereq_forest(Val(D); N=N)
    for (centering, G) in prereq_layouts(Val(D))
        fs = FieldSet{Float64}(forest, 1; G=G, centering=centering)
        c = staggers(fs)
        extent = ntuple(d -> size(fs.work, d), D)
        @test extent == ntuple(d -> N + 2 * G[d] + c[d], D)

        map_blocks!(prereq_stored_kernel!, fs, fs.work, Val(D); stored=true)
        wrong = 0
        for b in 1:nblocks(fs), I in CartesianIndices(extent)
            idx = Tuple(I)
            fs.work[idx..., 1, b] == prereq_tag(idx, b) || (wrong += 1)
        end
        @test wrong == 0

        # Said again where it bites, so that a failure names the ghosts
        # rather than a count: the low ghost plane of a dimension that has
        # one lies outside both the owned and the closed range.
        for d in 1:D
            G[d] == 0 && continue
            idx = ntuple(e -> e == d ? 1 : G[e] + 1, D)
            @test fs.work[idx..., 1, 1] == prereq_tag(idx, 1)
        end
    end

    # And the stored range is not the closed range asked for twice.
    fs = FieldSet{Float64}(forest, 1; G=2, centering=cellcentered(D))
    @test_throws "not both" map_blocks!(prereq_stored_kernel!, fs, fs.work, Val(D);
                                        stored=true, closed=true)
end

@testset "A field set is copied to the host without changing its layout: D=$D" for D in
                                                                                   PREREQ_DIMS
    # Guards `hostcopy`'s two halves. On the CPU it returns the field set
    # itself, so its copying path would otherwise be dead code on every
    # machine CI runs on; `hostcopy!` is that path, called here host to
    # host. The layout is the thing to get wrong: a destination built
    # without `G` and the centering has a differently shaped working array,
    # and a `copyto!` between two arrays of equal length and unequal shape
    # transposes the data instead of failing.
    forest = prereq_forest(Val(D))
    for (centering, G) in prereq_layouts(Val(D))
        src = FieldSet{Float64}(forest, D + 2; G=G, centering=centering)
        fill_by_coordinates!(AllVariables(prereq_state), src)
        @test hostcopy(src) === src

        dst = FieldSet{Float64}(src.forest, src.nvars; G=src.G,
                                centering=src.centering)
        @test TreeHydro.hostcopy!(dst, src) === dst
        @test dst.work !== src.work
        @test isequal(dst.work, src.work)
    end

    # A destination whose ghost width was left at something else is the
    # mistake, and it is an error rather than a transposition.
    src = FieldSet{Float64}(forest, D + 2; G=2, centering=cellcentered(D))
    thin = FieldSet{Float64}(forest, D + 2; G=1, centering=cellcentered(D))
    @test_throws "same layout" TreeHydro.hostcopy!(thin, src)
end

@testset "The resolved TreeAMR exports the M8 names the scheme needs" begin
    # A name list rather than a call: every one of these is reached for in
    # steps 1–9, and the cheapest place to find out that an upstream
    # release renamed one is here. `AllVariables` and the stored launch are exercised above;
    # these are the rest of the surface — the flux fixup, the
    # physical-boundary hook (this package is its first caller anywhere),
    # and the two reductions the criterion and the diagnostics go through.
    for name in (:AllVariables, :InterfaceSchedule, :restrict_interfaces!,
                 :CellBoundary, :boundary_by_coordinates, :firing_boxes,
                 :block_mapreduce)
        @test name in names(TreeAMR)
    end
end
