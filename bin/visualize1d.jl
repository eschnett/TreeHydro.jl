#!/usr/bin/env julia
#
# Visualize the tracked shock tube: the solution against Toro's exact
# Riemann solution, the refinement criterion that built the mesh under it,
# and what the conserved integrals did while it ran.
#
#     julia --project=bin bin/visualize1d.jl
#     julia --project=bin bin/visualize1d.jl --out=/tmp
#     julia --project=bin bin/visualize1d.jl --type=f32
#     julia --project=bin bin/visualize1d.jl --ops=1
#     julia --project=bin bin/visualize1d.jl --backend=metal --type=f32
#
# The figures are written as PNGs and, when stdout is a terminal, also
# drawn inline via SixelTerm. Piped or redirected output skips the inline
# draw rather than spraying sixel escapes into a log; `--display` and
# `--no-display` override the detection either way.
#
# There is no `--case=`: the tube is the only one-dimensional case that
# belongs here. The entropy wave is an exact solution the tests measure
# convergence against, and `CODE.md` says it does not appear in `bin/`.
#
# One figure, six panels. The left column is the final frame against `x`:
#
#   1. `ρ`, 2. `v`, 3. `p`, drawn one line per block and coloured by
#      refinement level, with the exact solution overlaid and the block
#      boundaries marked. Seeing *where* the coarse-fine faces sit relative
#      to the shock, the contact and the fan is the whole point of the
#      panel: the mesh is supposed to have put the finest blocks on the
#      three features and nowhere else;
#   4. the Löhner indicator τ per cell, with the refine and coarsen
#      thresholds drawn, so the mesh's own decision reads off the figure.
#      A captured shock sits near 0.57 at every spacing and never resolves
#      -- which is why `maxlevel_cap` and not the criterion is what stops
#      the refinement -- while the rarefaction's head falls with `h`.
#
# and the right column is against time:
#
#   5. the drift of each conserved integral from its initial value, on a
#      log axis. Mass and energy should sit at roundoff; the momentum
#      should sit on the closed-form boundary flux `(p_L - p_R)·t·A`
#      drawn beside it, because Sod's Dirichlet faces are a real flux and
#      not an error. `conserved_scales` gives the momentum *exactly zero*
#      on this case -- the gas starts at rest -- so this panel plots the
#      absolute drift and never divides by a scale;
#   6. the block count, with the level cap noted.
#
# The run comes from `evolve!` via its `observer` keyword, so this script
# contains no time-stepping loop of its own. See `CODE.md`.

using CairoMakie
using Printf
using SixelTerm
using TreeAMR
using TreeHydro

include(joinpath(@__DIR__, "backend.jl"))

# `type="png"` is not just a default worth keeping: CairoMakie's vector
# output turns every filled mark into its own polygon, and this figure
# draws one line per block over hundreds of blocks.
CairoMakie.activate!(; type="png", px_per_unit=2)

const LEVELCOLORS = Makie.wong_colors()

levelcolor(lvl) = LEVELCOLORS[mod1(lvl + 1, length(LEVELCOLORS))]

const FLOATTYPES = Dict("f32" => Float32, "f64" => Float64)

# Step 6's calibrated thresholds and step 7's driver configuration, quoted
# from `test/driver_tests.jl` rather than reinvented: the figure is meant
# to show the run the numbers in `CODE.md` were measured on.
const REFINE_TOL = 2 // 25            # 0.08
const COARSEN_TOL = 1 // 50           # 0.02
const SOD1D = (roots=(8,), N=8, cap=2, chunk=1 // 200, t_end=1 // 5)

"""
The conservative family at the interface order the scheme wants, as every
driver in the test suite builds it. `p` is the prolongation order, which
`--ops=` varies and nothing else does.
"""
viewer_ops(p=3) = Operators(family=Conservative, prolongation=p, restriction=2)

"""
One frame of a run: every block's primitives, its cell positions, its τ and
its level, plus the conserved totals and the block count the time series
needs.

Built inside the `observer` callback, where `U` is scattered and `P` is
current -- and *before* the regrid, which is the next thing the loop does
and which would invalidate both.

Everything kept here is **materialized**, and that is load-bearing rather
than tidy. `hostcopy` returns the field set *itself* on the CPU, by
design, and `regrid!` reuses and resizes the working array; a snapshot that
stored views would come back holding the last frame in every slot. The
`Float64.` conversions do the materializing and double as the one place a
`Float32` run stops being a `Float32` run -- everything below is a figure,
and Makie is happiest given `Float64`.

τ is computed here rather than asked of the package. `CODE.md` says as
much under "The refinement criterion": the criterion runs as a per-cell
predicate inside `firing_boxes`, which returns boxes and not values, so a
viewer that wants the value computes it, and `cell_tau` is exported for
exactly this.
"""
function snapshot(p, t; ε, ε_g)
    P = hostcopy(p.P)
    T = eltype(P.work)
    forest = P.forest
    N = forest.N
    G = P.G[1]
    scales = indicator_scales(P)
    refs = (T(scales[1]), T(scales[2]))
    εT, ε_gT = T(ε), T(ε_g)
    blocks = map(1:nblocks(P)) do b
        k = blockkey(P, b)
        ext = block_extent(forest, k)[1]
        (x=[Float64(coordinates(Float64, P, b, (i + G,))[1]) for i in 1:N],
         ρ=[Float64(P.work[i + G, 1, b]) for i in 1:N],
         v=[Float64(P.work[i + G, 2, b]) for i in 1:N],
         p=[Float64(P.work[i + G, 3, b]) for i in 1:N],
         τ=[Float64(cell_tau(P.work, (i + G,), b, refs, εT, ε_gT, Val(1)))
            for i in 1:N],
         ext=(Float64(ext[1]), Float64(ext[2])), lvl=level(k))
    end
    return (t=Float64(t), blocks=blocks, nblocks=nblocks(P),
            totals=map(Float64, conserved_totals(p.U)))
end

"""
Four round time ticks over `[0, t_end]`.

Spelled out because the right-hand column is narrow enough that Makie's
automatic choice draws labels on top of each other, which looks like a
rendering fault rather than a tick density.
"""
function timeticks(ts)
    hi = maximum(ts)
    vals = collect(range(0.0, hi; length=3))
    return (vals, [@sprintf("%.2f", v) for v in vals])
end

"""One panel of the left column: a field per block, coloured by level."""
function fieldpanel!(ax, blocks, field; exact=nothing)
    if exact !== nothing
        lines!(ax, exact[1], exact[2]; color=(:black, 0.55), linewidth=2.5,
               label="exact")
    end
    for b in blocks
        lines!(ax, b.x, getproperty(b, field); color=levelcolor(b.lvl),
               linewidth=1.4)
        # `CairoMakie.scatter!` spelled out: TreeAMR exports a `scatter!`
        # of its own -- the state-vector-into-field-set one -- and Makie
        # exports the plot recipe, so the bare name is ambiguous and errors
        # on use. The same trap is why nothing in this file ever writes
        # `density`: Makie's `@recipe` exports that too, and TreeHydro
        # exports it as a conserved-state accessor.
        CairoMakie.scatter!(ax, b.x, getproperty(b, field);
                            color=levelcolor(b.lvl), markersize=3)
    end
    # The block boundaries, which is where the coarse-fine faces are. Drawn
    # under the data rather than over it, and only at the finest level's
    # edges would be a different (and less honest) picture -- every block
    # edge is a face the fixup may have had to act on.
    vlines!(ax, unique(vcat([b.ext[1] for b in blocks],
                            [b.ext[2] for b in blocks]));
            color=(:grey, 0.28), linewidth=0.5)
    return ax
end

"""
The exact solution sampled on a dense grid, as `(x, ρ), (x, v), (x, p)`.

`sample` takes the similarity variable `ξ = (x - x₀)/t`, which is the whole
of what makes the exact solution cheap: it is self-similar, so one dense
sweep at the final time costs nothing and needs no mesh.
"""
function exactcurves(w, t; m=4000)
    sol = exact_riemann(w)
    L = Float64(TreeHydro.tofloat64(w.L))
    x₀ = Float64(TreeHydro.tofloat64(w.x₀))
    xs = range(0.0, L; length=m)
    samples = [sample(sol, (x - x₀) / Float64(t)) for x in xs]
    return (ρ=(xs, [s[1] for s in samples]), v=(xs, [s[2] for s in samples]),
            p=(xs, [s[3] for s in samples]))
end

"""
The figure: three primitives and τ against `x`, the drift and the block
count against `t`.
"""
function makefigure(snaps, w, title; tols, cap, area, ulp)
    last = snaps[end]
    exact = exactcurves(w, last.t)
    fig = Figure(; size=(1500, 1100))
    Label(fig[0, 1:2], title; fontsize=15, padding=(0, 0, 8, 0))

    axρ = Axis(fig[1, 1]; ylabel="ρ")
    axv = Axis(fig[2, 1]; ylabel="v")
    axp = Axis(fig[3, 1]; ylabel="p")
    axτ = Axis(fig[4, 1]; ylabel="τ", xlabel="x")
    fieldpanel!(axρ, last.blocks, :ρ; exact=exact.ρ)
    fieldpanel!(axv, last.blocks, :v; exact=exact.v)
    fieldpanel!(axp, last.blocks, :p; exact=exact.p)
    fieldpanel!(axτ, last.blocks, :τ)
    hlines!(axτ, [Float64(tols[1])]; color=:firebrick, linestyle=:dash,
            linewidth=1.5)
    hlines!(axτ, [Float64(tols[2])]; color=:steelblue, linestyle=:dot,
            linewidth=1.5)
    text!(axτ, 0.01, Float64(tols[1]); text=" refine", space=:relative,
          align=(:left, :bottom), color=:firebrick, fontsize=10)
    for a in (axρ, axv, axp)
        hidexdecorations!(a; grid=false)
    end
    linkxaxes!(axρ, axv, axp, axτ)

    ts = [s.t for s in snaps]
    totals0 = snaps[1].totals
    nvars = length(totals0)
    # Absolute drift, never relative: `conserved_scales` gives Sod's
    # momentum exactly zero because the gas starts at rest, so the obvious
    # normalization divides by zero on this very case. The log axis needs a
    # positive floor, and an exactly-zero drift -- which the first frame's
    # always is -- is drawn at it rather than dropped.
    #
    # `ulp` is the **run's** `eps` and not `eps(Float64)`, which matters and
    # is not cosmetic: at `Float32` many of these drifts are exactly zero,
    # and clipping them to a `Float64` floor would draw them eight orders
    # below the precision the run actually carries -- a figure claiming a
    # roundoff the arithmetic never had.
    driftfloor = ulp * maximum(abs, totals0)
    # Explicit ticks: this column is narrow, and Makie's automatic choice
    # overlaps its own labels at this width.
    tticks = timeticks(ts)
    axd = Axis(fig[1:2, 2]; yscale=log10, xlabel="t",
               ylabel="|total(t) − total(0)|", xticks=tticks,
               title="conserved integrals, absolute")
    names = ["mass"; ["S_$d" for d in 1:(nvars - 2)]; "energy"]
    handles = []
    for v in 1:nvars
        ys = [max(abs(s.totals[v] - totals0[v]), driftfloor) for s in snaps]
        push!(handles,
              lines!(axd, ts, ys;
                     color=LEVELCOLORS[mod1(v, length(LEVELCOLORS))],
                     linewidth=2))
    end
    # The momentum's yardstick, and it is a physical flux rather than an
    # error: the tube's Dirichlet faces hold `p_L` and `p_R` forever, so the
    # box gains momentum at exactly `(p_L − p_R)·A` per unit time. A run
    # whose momentum line sits on this one is conserving, not leaking.
    dp = Float64(TreeHydro.tofloat64(w.p_L)) - Float64(TreeHydro.tofloat64(w.p_R))
    push!(handles,
          lines!(axd, ts, [max(abs(dp * t * area), driftfloor) for t in ts];
                 color=:black, linestyle=:dash, linewidth=2))
    push!(handles,
          hlines!(axd, [driftfloor]; color=(:grey, 0.6), linestyle=:dot,
                  linewidth=1.5))
    push!(names, "(p_L − p_R)·t·A", "1 ulp of the largest total")

    axn = Axis(fig[3, 2]; xlabel="t", ylabel="blocks", xticks=tticks,
               title="mesh size (cap $cap)")
    lines!(axn, ts, [Float64(s.nblocks) for s in snaps]; color=:seagreen,
           linewidth=2)

    # Both legends live outside their axes, stacked in the last cell of the
    # right-hand column. Inside, either one covers data: this panel is
    # narrow and its legend is nearly as wide, so the momentum's steep rise
    # at `t = 0` is under any left placement and the roundoff floor is under
    # any bottom one.
    levels = sort(unique(b.lvl for b in last.blocks))
    legends = GridLayout(fig[4, 2])
    Legend(legends[1, 1], handles, names, "conserved integrals";
           framevisible=false, labelsize=9, titlesize=11, patchsize=(14, 2),
           tellheight=false, valign=:top)
    Legend(legends[2, 1],
           [LineElement(; color=levelcolor(l), linewidth=2) for l in levels],
           ["level $l" for l in levels], "refinement";
           framevisible=false, labelsize=9, titlesize=11,
           tellheight=false, valign=:top)

    colsize!(fig.layout, 1, Relative(0.70))
    for r in 1:4
        rowsize!(fig.layout, r, Relative(0.235))
    end
    rowgap!(fig.layout, 6)
    return fig
end

"""
The tracked shock tube: the adaptive run every claim in `CODE.md`'s
"Step 7" is about, at the configuration the tests measure it on.
"""
function sodcase(::Type{T}=Float64; ops_order=3, backend=CPU()) where {T}
    w = SodTube(T, Val(1))
    case = HydroCase(w; roots=SOD1D.roots)
    snaps = []
    observer(p, t, u) = push!(snaps, snapshot(p, t; ε=T(1 // 100),
                                              ε_g=T(1 // 1000)))
    r = evolve!(case, Val(1); N=SOD1D.N, ops=viewer_ops(ops_order),
                t_end=SOD1D.t_end, chunk=SOD1D.chunk, limiter=:minmod,
                refine_tol=REFINE_TOL, coarsen_tol=COARSEN_TOL,
                maxlevel_cap=SOD1D.cap, backend=backend, observer=observer)
    # The tube's Dirichlet faces are what the momentum yardstick needs the
    # area of. In one dimension the transverse extents are an empty product
    # and it is exactly 1, but writing it out is what makes the same line
    # right if this ever grows a planar sibling.
    area = prod(Float64(e[2] - e[1]) for e in case.extents[2:end]; init=1.0)
    title = @sprintf("Sod shock tube, %s, tracked at cap %d — %d cells in %d \
                      blocks, %d steps over %d chunks and %d regrids\n\
                      L1 against the exact solution %.4g, tracking %.3g, \
                      floor hits %d owned / %d ghost",
                     T, SOD1D.cap, r.cells, r.nblocks, r.nsteps, r.nchunks,
                     r.nregrids, r.l1, r.tracking, r.floor_hits, r.ghost_hits)
    return (snaps=snaps, w=w, r=r, title=title, area=area, ulp=eps(T))
end

function main(args)
    outdir = joinpath(@__DIR__, "output")
    ops_order = 3
    T = Float64
    typetag = ""
    backendname = "cpu"
    # Sixel is for a human looking at a terminal; a pipe gets the paths.
    inline = stdout isa Base.TTY
    for a in args
        if startswith(a, "--out=")
            outdir = a[7:end]
        elseif startswith(a, "--ops=")
            ops_order = parse(Int, a[7:end])
        elseif startswith(a, "--type=")
            tag = a[8:end]
            haskey(FLOATTYPES, tag) ||
                error("--type must be f32 or f64; got $tag")
            T = FLOATTYPES[tag]
            # The default type keeps the plain filename, so a Float32 render
            # never overwrites the figure CI checks.
            typetag = tag == "f64" ? "" : "_$tag"
        elseif startswith(a, "--backend=")
            backendname = a[11:end]
        elseif a == "--display"
            inline = true
        elseif a == "--no-display"
            inline = false
        else
            error("unknown argument $a; expected --out=, --ops=, --type=, \
                   --backend=, --display, --no-display")
        end
    end

    mkpath(outdir)
    @info "running the tracked Sod tube"
    # The run is what touches the storage; `withbackend` is what makes a
    # device package loaded a moment ago visible to it. Everything below is
    # a figure, drawn from the host copies `snapshot` already took.
    c = withbackend(backendname, T) do backend
        sodcase(T; ops_order=ops_order, backend=backend)
    end
    fig = makefigure(c.snaps, c.w, c.title; tols=(REFINE_TOL, COARSEN_TOL),
                     cap=SOD1D.cap, area=c.area, ulp=c.ulp)
    path = joinpath(outdir, "sod_1d" * typetag * ".png")
    save(path, fig)
    inline && display(fig)
    println("wrote $path")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
