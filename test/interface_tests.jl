# Coarse-fine faces: the mesh before anything runs on it.
#
# The claim the package exists to make — that TreeAMR's interface flux
# restriction is what conserves across a refinement boundary, and that the
# prolongation order is what caps the rate in L∞ — is measured in
# `test/long/interface_tests.jl`, which is minutes of evolution and belongs
# to the long tier. What stays here is the one testset that costs
# milliseconds and that every one of those claims rests on: that the
# two-level Sod forests really are two-level.
#
# It is not decoration. Every claim next door is about a coarse-fine face,
# and a forest that quietly came out single-level would make all of them
# pass for the wrong reason — a conservation test on a mesh with no
# coarse-fine face in it passes whatever the fixup does. Asserting the mesh
# separately, from the block extents rather than from a run, is what makes
# the expensive claims mean something, and it is why this half is cheap
# enough to run on every push.

@testset "The two-level Sod forests are the meshes they claim to be" begin
    # The mesh before anything is run on it, because every claim below is
    # about a coarse-fine face and a forest that quietly came out
    # single-level would make all of them pass for the wrong reason. Three
    # configurations, and the two ways of asking for one that does not
    # exist.
    T = Float64
    for (D, roots) in ((1, (4,)), (2, (4, 1)))
        uniform = sod_forest(Val(D), 8; roots=roots)
        @test forest_levels(uniform) == [0]
        @test nleaves(uniform) == prod(roots)
        for refined in (:middle, :left)
            f = sod_forest(Val(D), 8; roots=roots, refined=refined)
            @test forest_levels(f) == [0, 1]
            @test nleaves(f) > nleaves(uniform)
            # Two of the four root blocks along the tube are refined in
            # both configurations — the middle two and the left two — so
            # the two meshes have the same size and differ only in where
            # the fine half sits.
            @test nleaves(f) == prod(roots) - 2 + 2 * 2^D
        end
        # `:none` and `false` are the same mesh, and the Dirichlet face is
        # covered by level-1 blocks under `:left` and level-0 ones under
        # `:middle`. Asserted through the block extents rather than through
        # a run, so a change to the criterion shows here first.
        @test nleaves(sod_forest(Val(D), 8; roots=roots, refined=:none)) ==
              nleaves(uniform)
        for (refined, lvl) in ((:middle, 0), (:left, 1))
            f = sod_forest(Val(D), 8; roots=roots, refined=refined)
            low = filter(k -> block_extent(f, k)[1][1] == 0, f.leaves)
            @test !isempty(low)
            @test all(k -> level(k) == lvl, low)
        end
    end
    # A misspelt configuration is refused, and so is one that selects no
    # block: with two roots along the tube neither center lies inside
    # `(L/4, 3L/4)`, and the mesh would come out single-level.
    @test_throws "must be false, :none, :middle or :left" sod_forest(
        Val(1), 8; roots=(4,), refined=:right)
    @test_throws "selected no root block to refine" sod_forest(
        Val(1), 8; roots=(2,), refined=:middle)
end
