#!/usr/bin/env julia
#
# Visualize the two-dimensional cases: the Kelvin-Helmholtz shear layer
# whose feature *grows*, and the Sedov blast whose feature *travels*.
#
#     julia --project=bin bin/visualize2d.jl
#     julia --project=bin bin/visualize2d.jl --case=kh
#     julia --project=bin bin/visualize2d.jl --case=sedov --out=/tmp
#     julia --project=bin bin/visualize2d.jl --case=kh --movie
#     julia --project=bin bin/visualize2d.jl --movie --movie-frames=60 --movie-format=gif
#     julia --project=bin bin/visualize2d.jl --type=f32
#     julia --project=bin bin/visualize2d.jl --backend=metal --type=f32
#
# A separate script from `visualize1d.jl` and not a `--dim=2` flag on it,
# because nothing transfers: that viewer draws one line per block against
# x, which in 2D is not a worse picture but no picture at all. `CODE.md`
# says as much -- a higher dimension needs a different figure, not a
# different argument.
#
# Both cases share a filmstrip and differ in what goes under it.
#
# `--case=kh`, three parts:
#
#   1. a filmstrip of `ρ` at four times, one heatmap per block with the
#      block boundaries drawn and coloured by refinement level. The colour
#      range is fixed across the four, so the strip shows the layer rolling
#      up rather than four separate renormalizations;
#   2. `M(t)` and the maximum `y`-kinetic energy, McNally's two
#      diagnostics, with the uniform *fine* run over them -- which is the
#      only quantitative reference this case has, there being no closed
#      form. The band is the window `growth_rate` fits over, `2a ≤ M ≤ 6a`;
#   3. the block count against time, with the uniform fine mesh's drawn
#      across it. The gap between them is the saving, and this is the one
#      case in the package whose refined region *grows*.
#
# `--case=sedov`, three parts:
#
#   1. the same filmstrip;
#   2. `ρ` against radius for every owned cell of the final frame. The
#      exact solution is a function of `r` alone, so every cell of a
#      perfect answer lands on one curve; vertical spread is what the
#      Cartesian mesh costs, and a coarse-fine artifact shows up as a band
#      rather than a curve. The similarity profile is drawn over it against
#      the **measured** `E₀` and stops at the shock, the law being the
#      strong-shock limit and describing nothing outside it. Note that this
#      box is Dirichlet on every face, so -- unlike TreeWave's periodic
#      blast -- there is no `L/2` rule and the scatter's outer tail is
#      simply undisturbed ambient in the corners;
#   3. the shock radius against time on log axes with the similarity law
#      over it, and the block count beside it.
#
# `--movie` writes a video beside the figure, from the same run: every
# frame the observer hands over rather than the four the filmstrip draws,
# which for the shear layer is 301 of them and a ten-second animation. It
# is what the strip cannot show -- the *order* in which things happen, the
# mode decaying before it grows, the refined region thickening with the
# rolls rather than travelling with them. `--movie-frames=` shortens it,
# `--movie-fps=` sets the rate and `--movie-format=gif` changes the
# container; `FFMPEG_jll` comes with Makie, so none of this is a new
# dependency. The figure is byte-identical whether or not a movie is asked
# for, which is what `keptframes` is for.
#
# The runs come from `kh_run` and `evolve!` via their `observer` keyword,
# so this script contains no time-stepping loop of its own. See `CODE.md`.

using CairoMakie
using Printf
using SixelTerm
using TreeAMR
using TreeHydro

include(joinpath(@__DIR__, "backend.jl"))

# `type="png"` is not just a default worth keeping. CairoMakie's fast image
# path -- the one that pads each heatmap's edges so that abutting blocks
# leave no hairline seam -- is disabled for vector output, where every
# heatmap instead degrades into one polygon per cell. A few hundred blocks
# of 8x8 is tens of thousands of those.
CairoMakie.activate!(; type="png", px_per_unit=2)

const LEVELCOLORS = Makie.wong_colors()

levelcolor(lvl) = LEVELCOLORS[mod1(lvl + 1, length(LEVELCOLORS))]

const FLOATTYPES = Dict("f32" => Float32, "f64" => Float64)

# Steps 6, 9 and 10's calibrated thresholds and configurations, quoted from
# `test/kelvinhelmholtz_tests.jl` and `test/sedov_tests.jl` rather than
# reinvented: the figures are meant to show the runs the numbers in
# `CODE.md` were measured on.
const REFINE_TOL = 2 // 25            # 0.08
const COARSEN_TOL = 1 // 50           # 0.02
# The same colormap each case's filmstrip uses, so the movie and the figure
# are the same picture in motion and not two different ones.
const MOVIECOLORMAPS = Dict("kh" => :viridis, "sedov" => :inferno)

const KH = (roots=4, N=8, cap=2, chunk=1 // 200, t_end=3 // 2)
const SEDOV2D = (roots=4, N=8, cap=2, r₀=1 // 16, chunk=1 // 400, t_end=1 // 10)

viewer_ops(p=3) = Operators(family=Conservative, prolongation=p, restriction=2)

"""
One frame: every block's `ρ`, the geometry a heatmap needs, and its level.

Built inside the `observer` callback, where `U` is scattered and `P` is
current and *before* the regrid that would invalidate both.

`Float64.(interiorview(...))` rather than the view itself, and that is
load-bearing rather than tidy: `hostcopy` returns the field set **itself**
on the CPU by design, and `regrid!` reuses and resizes the working array,
so a frame that stored a view would come back holding the last frame. The
conversion does the materializing and doubles as the one place a `Float32`
run stops being one -- everything below is a figure, and Makie is happiest
given `Float64`.

Note `interiorview(P, b, 1)` gives `ρ[i,j]` at `(x_i, y_j)`, which is
already `heatmap!`'s convention. **Not transposed**: on the shear layer,
which is banded in `y`, a transpose is obvious; on a radial blast it is not
obvious at all, which is what makes it worth saying.
"""
function frame2d(p, t)
    P = hostcopy(p.P)
    forest = P.forest
    N = forest.N
    blocks = map(1:nblocks(P)) do b
        k = blockkey(P, b)
        # Where the samples are, asked of the field set rather than
        # reconstructed: `coordinates` takes **stored** indices, so the
        # first owned cell is at `G[d] + 1`. Both the heatmap's edges and
        # the radial scatter derive from this pair and nothing else.
        first = coordinates(Float64, P, b, ntuple(d -> P.G[d] + 1, 2))
        (ext=map(e -> (Float64(e[1]), Float64(e[2])), block_extent(forest, k)),
         first=first, h=Float64(spacing(Float64, forest, k)),
         ρ=Float64.(interiorview(P, b, 1)), lvl=level(k))
    end
    return (t=Float64(t), blocks=blocks, nblocks=nblocks(P), N=N)
end

"""
The `N + 1` cell **edges** of a block along dimension `d`.

The edges and not a bare `(lo, hi)` tuple, and that is load-bearing:
`heatmap!` reads a two-element range as the *centres* of the first and last
cell, which inflates every block by `w/2(N-1)` per side -- 7% at `N = 8` --
so the blocks silently overlap and the mesh looks subtly wrong rather than
broken.

Derived from the samples and the spacing rather than from the extent, so
that there is one place the geometry is read. The two agree here because
this package's fields are cell-centered in every dimension and refuse to be
anything else, which is a simplification over TreeWave's viewers, where a
vertex-centred block's samples sit half a spacing off its outlines.
"""
edges(b, d, N) = range(b.first[d] - b.h / 2, b.first[d] + (N - 1 // 2) * b.h;
                       length=N + 1)

"""One filmstrip panel: the field as one heatmap per block."""
function fieldpanel!(ax, snap; colorrange, colormap)
    local hm
    for b in snap.blocks
        hm = heatmap!(ax, edges(b, 1, snap.N), edges(b, 2, snap.N), b.ρ;
                      colormap=colormap, colorrange=colorrange)
    end
    return hm
end

"""
The block boundaries, one `lines!` per level rather than one per block.

`NaN` breaks the polyline between rectangles, which is what lets a whole
level go in one plot object. Makie costs on the order of a millisecond per
object, and a few hundred blocks over four frames is otherwise seconds of
figure time for a picture that draws in one.
"""
function blockoutlines!(ax, snap)
    for lvl in sort(unique(b.lvl for b in snap.blocks))
        pts = Point2f[]
        for b in snap.blocks
            b.lvl == lvl || continue
            (x0, x1), (y0, y1) = b.ext[1], b.ext[2]
            append!(pts, [Point2f(x0, y0), Point2f(x1, y0), Point2f(x1, y1),
                          Point2f(x0, y1), Point2f(x0, y0),
                          Point2f(NaN, NaN)])
        end
        lines!(ax, pts; color=levelcolor(lvl), linewidth=0.6)
    end
    return ax
end

"""The four frames laid out with a shared colour range and one colorbar."""
function filmstrip!(layout, snaps; colormap)
    lo = minimum(minimum(minimum(b.ρ) for b in s.blocks) for s in snaps)
    hi = maximum(maximum(maximum(b.ρ) for b in s.blocks) for s in snaps)
    local hm
    for (k, s) in enumerate(snaps)
        ax = Axis(layout[1, k]; aspect=DataAspect(), titlesize=11,
                  title=@sprintf("t = %.3f, %d blocks", s.t, s.nblocks))
        hidedecorations!(ax)
        hm = fieldpanel!(ax, s; colorrange=(lo, hi), colormap=colormap)
        blockoutlines!(ax, s)
    end
    Colorbar(layout[1, length(snaps) + 1], hm; label="ρ", width=12,
             height=Relative(0.9))
    colgap!(layout, 6)
    return layout
end

"""
    moviefile(path, snaps; colormap, fps, label) -> path

Every retained frame as a video, one heatmap per block with the block
outlines over it -- the filmstrip's panel, animated.

This is what the four-frame strip cannot show: the shear layer *evolves*,
and its whole interest is the order in which things happen. The seeded mode
decays while the ramp sheds its transient, takes off around `t = 0.5`, and
rolls up; the refined region thickens with the rolls rather than travelling
with them. Four stills sample that; 301 frames are the thing itself.

Three details that are not free choices:

- **`empty!(ax)` every frame.** [`fieldpanel!`](@ref) emits one `heatmap!`
  per block and the mesh is not fixed -- the shear layer grows 160 -> 232
  blocks over three regrids -- so a frame drawn on top of the last would
  keep the previous mesh's blocks underneath it wherever the new one is
  coarser.
- **The colour range is fixed over the whole movie**, taken from the frames
  that will actually be drawn, exactly as [`filmstrip!`](@ref) takes it. A
  per-frame range would renormalize every frame and turn a growing
  instability into a constant-looking one.
- **The `Colorbar` is built from `colorrange` and not from a plot handle.**
  The handle a `heatmap!` returns is deleted by the next `empty!`, so a
  colorbar attached to one would be pointing at a dead plot from frame two
  onward.

`Makie.record` picks the container from the extension, and both `.mp4` and
`.gif` work here: `FFMPEG_jll` arrives as a dependency of Makie, so the
movie costs `bin/Project.toml` nothing.
"""
function moviefile(path, snaps; colormap, fps, label)
    lo = minimum(minimum(minimum(b.ρ) for b in s.blocks) for s in snaps)
    hi = maximum(maximum(maximum(b.ρ) for b in s.blocks) for s in snaps)
    fig = Figure(; size=(900, 950))
    ax = Axis(fig[1, 1]; aspect=DataAspect(), titlesize=13)
    hidedecorations!(ax)
    Colorbar(fig[1, 2]; colorrange=(lo, hi), colormap=colormap, label="ρ",
             width=12, height=Relative(0.9))
    record(fig, path, eachindex(snaps); framerate=fps) do k
        s = snaps[k]
        empty!(ax)
        fieldpanel!(ax, s; colorrange=(lo, hi), colormap=colormap)
        blockoutlines!(ax, s)
        ax.title = @sprintf("%s — t = %.3f, %d blocks", label, s.t, s.nblocks)
    end
    return path
end

"""
Which observer calls to keep a frame from: `n` of them, evenly spaced over
the run and always including the first and the last.

Two consumers want different answers, which is why this is a function and
not a constant. The figure wants four; the movie wants all of them, or
`--movie-frames=` of them. Selectivity is what makes it necessary rather
than tidy: the shear layer takes 301 observer calls over a few hundred
blocks of 64 cells, which is tens of megabytes of `ρ` for a figure that
draws four.

[`keptframes`](@ref) takes the **union** of the two, so that asking for a
movie never changes which four frames the figure draws.
"""
framepicks(ncalls, n=4) = Set(round.(Int, range(1, ncalls; length=n)))

"""
    keptframes(ncalls; movie, movie_frames) -> (keep, film, mov)

The observer calls worth a frame, split by who wants them.

`film` is the figure's four and is **not** a function of the movie
settings, which is the whole point: `keep` is the union, so a run with
`--movie` retains more frames but draws the identical filmstrip from the
identical four. The figure a viewer gets is the same file either way, and
the test for that is `cmp`, not inspection.
"""
function keptframes(ncalls; movie::Bool, movie_frames)
    film = framepicks(ncalls, 4)
    n = movie_frames === nothing ? ncalls : min(movie_frames, ncalls)
    mov = movie ? framepicks(ncalls, n) : Set{Int}()
    return (union(film, mov), film, mov)
end

"""
Split what the observer kept into the figure's frames and the movie's.

The frames are stored as `(call, frame)` pairs so that this selection is by
the observer call they came from rather than by position, which is what
keeps the two independent of each other.
"""
filmframes(kept, film) = [f for (i, f) in kept if i in film]
movieframes(kept, mov) = [f for (i, f) in kept if i in mov]

# --------------------------------------------------------------------------
# Kelvin-Helmholtz
# --------------------------------------------------------------------------

"""
The shear layer, tracked, beside the uniform fine run it is judged against.

The frames come through [`kh_run`](@ref)'s `observer` pass-through, so
`M(t)` and the kinetic energy on the figure are the curves the case's own
diagnostics recorded and not a second computation of them -- which is the
whole reason that keyword exists. See "Step 10" in `CODE.md`.
"""
function khcase(::Type{T}=Float64; ops_order=3, backend=CPU(), movie=false,
                movie_frames=nothing) where {T}
    ncalls = Int(ceil(KH.t_end / KH.chunk)) + 1
    keep, film, mov = keptframes(ncalls; movie=movie, movie_frames=movie_frames)
    kept, seen = Tuple{Int,Any}[], Ref(0)
    function grab(p, t, u)
        seen[] += 1
        seen[] in keep && push!(kept, (seen[], frame2d(p, t)))
    end
    @info "running the tracked shear layer"
    tr = kh_run(T, Val(2); N=KH.N, ops=viewer_ops(ops_order), chunk=KH.chunk,
                maxlevel_cap=KH.cap, refine_tol=REFINE_TOL,
                coarsen_tol=COARSEN_TOL, t_end=KH.t_end, roots=KH.roots,
                riemann=:hllc, backend=backend, observer=grab)
    @info "running the uniform fine reference"
    # `scale = 2^cap` shares the tracked run's *finest* spacing, and the
    # same `chunk` makes the two sample `M(t)` at the same times -- which is
    # what makes the two curves comparable at all.
    fine = kh_uniform(T, Val(2); N=KH.N, ops=viewer_ops(ops_order),
                      chunk=KH.chunk, t_end=KH.t_end, roots=KH.roots,
                      scale=2^KH.cap, riemann=:hllc, backend=backend)
    # The fit window is a range of `M` and not of `t`, so that it means the
    # same phase of the instability at every resolution: `2a ≤ M ≤ 6a` on the
    # seeded amplitude `a`, which is `test/kelvinhelmholtz_tests.jl`'s
    # `KH_WINDOW` and the window every rate in `CODE.md` was fitted over.
    a = Float64(TreeHydro.tofloat64(tr.w.a))
    rate = growth_rate(tr.ts, tr.Ms; from=2a, to=6a)
    title = @sprintf("Kelvin–Helmholtz shear layer, %s, HLLC — tracked at cap \
                      %d, %d cells in %d blocks against %d uniformly fine\n\
                      M(0) = %.4f grows to M(%.2f) = %.5f at a fitted rate of \
                      %.5f, below the incompressible bounds 4.384 and 5.9238",
                     T, KH.cap, tr.r.cells, tr.r.nblocks, fine.r.cells,
                     tr.Ms[1], tr.ts[end], tr.Ms[end], rate)
    return (snaps=filmframes(kept, film), movie=movieframes(kept, mov),
            tr=tr, fine=fine, title=title,
            label=@sprintf("Kelvin–Helmholtz, %s, HLLC, cap %d", T, KH.cap))
end

function khfigure(c)
    fig = Figure(; size=(1400, 1150))
    Label(fig[0, 1:2], c.title; fontsize=15, padding=(0, 0, 8, 0))
    filmstrip!(GridLayout(fig[1, 1:2]), c.snaps; colormap=:viridis)

    tr, fine = c.tr, c.fine
    a = Float64(TreeHydro.tofloat64(tr.w.a))
    # The two diagnostics side by side in a layout of their own, so that
    # neither is squeezed into the legend column's width.
    series = GridLayout(fig[2, 1:2])
    axM = Axis(series[1, 1]; yscale=log10, xlabel="t", ylabel="M(t)",
               title="the seeded mode, with the fit window shaded")
    # The window `growth_rate` fits over, drawn rather than described: the
    # rate is a property of this band and of nothing outside it.
    hspan!(axM, 2a, 6a; color=(:steelblue, 0.10))
    lines!(axM, tr.ts, tr.Ms; linewidth=2, label="tracked, cap $(KH.cap)")
    lines!(axM, fine.ts, fine.Ms; linewidth=2, linestyle=:dash, color=:black,
           label="uniform fine")
    axislegend(axM; position=:rb, framevisible=false, labelsize=10)

    axK = Axis(series[1, 2]; yscale=log10, xlabel="t",
               ylabel="max ½ρv_y²",
               title="the kinetic-energy diagnostic")
    lines!(axK, tr.ts, tr.Ks; linewidth=2)
    lines!(axK, fine.ts, fine.Ks; linewidth=2, linestyle=:dash, color=:black)

    axn = Axis(fig[3, 1]; xlabel="t", ylabel="blocks",
               title="the refined region grows with the rolls")
    lines!(axn, tr.ts, Float64.(tr.nbs); color=:seagreen, linewidth=2,
           label="tracked")
    hlines!(axn, [Float64(fine.nbs[1])]; color=:black, linestyle=:dash,
            linewidth=2, label="uniform fine")
    axislegend(axn; position=:rb, framevisible=false, labelsize=10)

    levels = sort(unique(b.lvl for b in c.snaps[end].blocks))
    Legend(fig[3, 2],
           [LineElement(; color=levelcolor(l), linewidth=2) for l in levels],
           ["level $l" for l in levels], "refinement";
           framevisible=false, tellheight=false, valign=:top)
    rowsize!(fig.layout, 1, Aspect(1, 0.27))
    rowsize!(fig.layout, 3, Relative(0.22))
    rowgap!(fig.layout, 8)
    return fig
end

# --------------------------------------------------------------------------
# Sedov
# --------------------------------------------------------------------------

"""
The blast, tracked, with the shock radius and the peak compression recorded
per chunk through the observer -- which is the only way to take them, the
state being scattered and `P` current exactly there.

There is no `sedov_run` in `src/`: `sedov_static` is the *static*-mesh
measurement driver and takes no observer, so the tracked run is assembled
here as `test/sedov_tests.jl` assembles it.
"""
function sedovcase(::Type{T}=Float64; ops_order=3, backend=CPU(), movie=false,
                   movie_frames=nothing) where {T}
    w = SedovBlast(T, Val(2); r₀=SEDOV2D.r₀)
    case = HydroCase(w; roots=SEDOV2D.roots)
    ncalls = Int(ceil(SEDOV2D.t_end / SEDOV2D.chunk)) + 1
    keep, film, mov = keptframes(ncalls; movie=movie, movie_frames=movie_frames)
    kept, seen = Tuple{Int,Any}[], Ref(0)
    ts, rs, peaks, nbs = Float64[], Float64[], Float64[], Int[]
    function watch(p, t, u)
        seen[] += 1
        push!(ts, Float64(t))
        push!(rs, Float64(shock_radius(p.P, w)))
        push!(peaks, Float64(peak_compression(p.P, w)))
        push!(nbs, nblocks(p.P))
        seen[] in keep && push!(kept, (seen[], frame2d(p, t)))
    end
    @info "running the tracked blast"
    r = evolve!(case, Val(2); N=SEDOV2D.N, ops=viewer_ops(ops_order),
                t_end=SEDOV2D.t_end, chunk=SEDOV2D.chunk, limiter=:minmod,
                refine_tol=REFINE_TOL, coarsen_tol=COARSEN_TOL,
                maxlevel_cap=SEDOV2D.cap, accounting=true, backend=backend,
                observer=watch)
    # The *measured* energy and not the nominal: which cell centres fall
    # inside `r₀` is a property of the mesh, and the ratio is 1.0345 in
    # `D = 2`. `measured_E₀` reads the run's own initial totals, so it can
    # only be taken after the run.
    E₀ = measured_E₀(r, w)
    sim = sedov_similarity(w)
    # The floor counts are expected to be **zero** here, and saying so is the
    # point rather than an omission: tracking puts the refined region's
    # boundary ahead of the shock, so a tracked run's coarse-fine faces stand
    # in undisturbed ambient and nothing is floored. What fires the pressure
    # floor on this case is a *static* mesh the blast crosses --
    # `sedov_forest(:center)`, where the tests measure it. See "Step 9" in
    # `CODE.md`.
    title = @sprintf("Sedov blast, %s, D = 2 — tracked at cap %d, %d cells in \
                      %d blocks, %d steps\nmeasured E₀ = %.6g (nominal %.6g), \
                      exponent %.5f against 1/2, peak compression %.4f against \
                      the strong-shock 6\nfloor hits %d owned / %d ghost — \
                      tracking keeps every coarse-fine face in undisturbed \
                      ambient, which is why a tracked mesh cannot measure them",
                     T, SEDOV2D.cap, r.cells, r.nblocks, r.nsteps, E₀,
                     Float64(TreeHydro.tofloat64(w.E₀)),
                     exponent_fit(ts, rs; from=3 * Float64(TreeHydro.tofloat64(w.r₀))),
                     peaks[end], r.floor_hits, r.ghost_hits)
    return (snaps=filmframes(kept, film), movie=movieframes(kept, mov),
            r=r, w=w, sim=sim, E₀=E₀, ts=ts, rs=rs, peaks=peaks, nbs=nbs,
            title=title,
            label=@sprintf("Sedov blast, %s, D = 2, cap %d", T, SEDOV2D.cap))
end

"""
The similarity profile as `(r, ρ)` at time `t`, for `r ≤ r_s`.

**Parametric in `u = V − V₀` and never in `λ`**: `sedov_profile`'s map
`u ↦ λ` is closed-form and inverting it would be a root find this package
does not do. `u` is sampled log-spaced on `(0, u₂]` because `λ → 0` as
`u → 0⁺`, and never *at* zero, where the profile is singular.
"""
function profilecurve(sim, t, E₀, ρ₀; m=600)
    us = exp.(range(log(sim.u₂ * 1e-6), log(sim.u₂); length=m))
    rs = sedov_radius(sim, t, E₀, ρ₀)
    pts = [sedov_profile(sim, u) for u in us]
    return ([p[1] * rs for p in pts], [p[2] * ρ₀ for p in pts], rs)
end

function sedovfigure(c)
    fig = Figure(; size=(1400, 1150))
    Label(fig[0, 1:2], c.title; fontsize=14, padding=(0, 0, 8, 0))
    filmstrip!(GridLayout(fig[1, 1:2]), c.snaps; colormap=:inferno)

    last = c.snaps[end]
    ρ₀ = Float64(TreeHydro.tofloat64(c.w.ρ₀))
    # Every owned cell as `(r, ρ)`, in ONE scatter call rather than one per
    # block: Makie's per-object cost makes the second form five times the
    # price for the identical picture. No minimum image -- this box is
    # Dirichlet on every face, so unlike TreeWave's periodic blast there
    # are no images and no `L/2` rule.
    #
    # `CairoMakie.scatter!` spelled out below, because TreeAMR exports a
    # `scatter!` of its own and the bare name is ambiguous. The same trap is
    # why nothing here writes `density`: Makie's `@recipe` exports it and so
    # does TreeHydro. The slot is read directly instead.
    radii, dens, cols = Float64[], Float64[], RGBAf[]
    for b in last.blocks
        col = levelcolor(b.lvl)
        N = last.N
        for j in 1:N, i in 1:N
            x = b.first[1] + (i - 1) * b.h
            y = b.first[2] + (j - 1) * b.h
            push!(radii, hypot(x, y))
            push!(dens, b.ρ[i, j])
            push!(cols, RGBAf(col.r, col.g, col.b, 0.35))
        end
    end
    axr = Axis(fig[2, 1:2]; xlabel="|x|", ylabel="ρ",
               title="ρ against radius, every owned cell of the final frame — \
                      one curve means the mesh kept the blast radial")
    CairoMakie.scatter!(axr, radii, dens; color=cols, markersize=2)
    pr, pρ, r_law = profilecurve(c.sim, c.ts[end], c.E₀, ρ₀)
    lines!(axr, pr, pρ; color=(:black, 0.85), linewidth=1.6, linestyle=:dash,
           label="similarity profile at the measured E₀")
    hlines!(axr, [ρ₀]; color=(:grey, 0.7), linestyle=:dot, linewidth=1.5,
            label="ambient")
    vlines!(axr, [r_law]; color=:black, linewidth=1, label="r_s (law)")
    vlines!(axr, [c.rs[end]]; color=:firebrick, linewidth=1,
            label="r_s (measured)")
    axislegend(axr; position=:rt, framevisible=false, labelsize=10)

    axs = Axis(fig[3, 1]; xscale=log10, yscale=log10, xlabel="t",
               ylabel="r_s", title="the shock radius against the law")
    keep = c.ts .> 0
    lines!(axs, c.ts[keep], c.rs[keep]; linewidth=2, label="measured")
    lines!(axs, c.ts[keep],
           [sedov_radius(c.sim, t, c.E₀, ρ₀) for t in c.ts[keep]];
           color=:black, linestyle=:dash, linewidth=2, label="similarity law")
    axislegend(axs; position=:rb, framevisible=false, labelsize=10)

    axn = Axis(fig[3, 2]; xlabel="t", ylabel="blocks",
               title="the refined region is a disk, not a shell")
    lines!(axn, c.ts, Float64.(c.nbs); color=:seagreen, linewidth=2)

    rowsize!(fig.layout, 1, Aspect(1, 0.27))
    rowgap!(fig.layout, 8)
    return fig
end

function main(args)
    case = "both"
    outdir = joinpath(@__DIR__, "output")
    ops_order = 3
    T = Float64
    typetag = ""
    backendname = "cpu"
    movie = false
    # `nothing` means every observer sample -- 301 on the shear layer, a
    # ten-second movie at the default rate. `--movie-frames=` is for the CI
    # smoke test and for anyone who wants a quicker look.
    movie_frames = nothing
    movie_fps = 30
    movie_ext = "mp4"
    # Sixel is for a human looking at a terminal; a pipe gets the paths.
    inline = stdout isa Base.TTY
    for a in args
        if startswith(a, "--case=")
            case = a[8:end]
        elseif startswith(a, "--out=")
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
        elseif a == "--movie"
            movie = true
        elseif startswith(a, "--movie-frames=")
            movie_frames = parse(Int, a[16:end])
            movie_frames ≥ 2 ||
                error("--movie-frames must be at least 2; got $movie_frames")
            movie = true
        elseif startswith(a, "--movie-fps=")
            movie_fps = parse(Int, a[13:end])
            movie_fps ≥ 1 || error("--movie-fps must be positive; got $movie_fps")
        elseif startswith(a, "--movie-format=")
            movie_ext = a[16:end]
            movie_ext in ("mp4", "gif") ||
                error("--movie-format must be mp4 or gif; got $movie_ext")
        elseif a == "--display"
            inline = true
        elseif a == "--no-display"
            inline = false
        else
            error("unknown argument $a; expected --case=, --out=, --ops=, \
                   --type=, --backend=, --movie, --movie-frames=, \
                   --movie-fps=, --movie-format=, --display, --no-display")
        end
    end
    case in ("both", "kh", "sedov") ||
        error("--case must be kh, sedov, or both; got $case")

    mkpath(outdir)
    # Everything that touches the storage runs inside `withbackend`, which
    # is what makes a device package loaded a moment ago visible to it.
    # Everything below is a figure, drawn from the host copies the frames
    # already took.
    written = withbackend(backendname, T) do backend
        paths = String[]
        for (name, build, draw) in (("kh", khcase, khfigure),
                                    ("sedov", sedovcase, sedovfigure))
            (case == "both" || case == name) || continue
            c = build(T; ops_order=ops_order, backend=backend, movie=movie,
                      movie_frames=movie_frames)
            fig = draw(c)
            path = joinpath(outdir, "$(name)_2d$(typetag).png")
            save(path, fig)
            # A still can go to the terminal; a video cannot, so `--display`
            # says nothing about the movie either way.
            inline && display(fig)
            push!(paths, path)
            if movie
                @info "encoding $(length(c.movie)) frames"
                push!(paths,
                      moviefile(joinpath(outdir,
                                         "$(name)_2d$(typetag).$(movie_ext)"),
                                c.movie; colormap=MOVIECOLORMAPS[name],
                                fps=movie_fps, label=c.label))
            end
        end
        paths
    end
    for p in written
        println("wrote $p")
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
