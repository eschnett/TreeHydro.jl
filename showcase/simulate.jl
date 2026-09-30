# The simulation half of the showcase: the reflecting Kelvin–Helmholtz half
# box on a mesh whose levels follow a zooming camera, writing one frame file
# per movie frame. `render.jl` turns the frames into a movie in a second,
# independent step.
#
#     julia -t auto --project=showcase showcase/simulate.jl \
#         --config=showcase/configs/pilot.toml --out=showcase/output/pilot
#
# Options: `--config=` (required), `--out=` (required), `--backend=cpu|cuda|
# metal`, `--restart=FILE` (a checkpoint this script wrote),
# `--stop-after=K`, which checkpoints after frame K and stops, and
# `--walltime-hours=H`, which overrides `run.walltime_hours`.
#
# The loop is `evolve!`'s (src/driver.jl), in the same order and from the
# same pieces — step, CFL recheck, observe, regrid, rebuild, reset — with the
# two things `evolve!` fixes for a whole run made per chunk: the chunk end
# times are the movie's frame times, which shrink as the zoom grows, and the
# level cap is per block, a window about the camera. See README.md.

using TreeAMR, TreeHydro, HDF5
using KernelAbstractions: KernelAbstractions, Backend, CPU, allocate, get_backend

include(joinpath(@__DIR__, "kh_zoom.jl"))
include(joinpath(@__DIR__, "..", "bin", "backend.jl"))

const IRK = TreeHydro.IRK
const T = Float64
const APPLICATION = "TreeHydroShowcase" => 1

# --- sampling the view -----------------------------------------------------

# Point buffers per size, on the host and on the backend, reused across frames:
# a production frame is 8.3M points.
const POINTBUFS = Dict{Tuple{Int,Any},Any}()

function pointbuffers(n, backend)
    get!(POINTBUFS, (n, backend)) do
        host = Vector{NTuple{2,T}}(undef, n)
        dev = backend isa CPU ? host : allocate(backend, NTuple{2,T}, n)
        (host, dev)
    end
end

"""
    sample_rect(P, C, Wx, Wy, nx, ny, backend) -> Matrix{T}

`ρ` on an `nx × ny` grid of points over the reconstructed rectangle of centre
`C` and size `Wx × Wy`, index `[i, j]` with `i` left to right and `j` bottom
to top, by TreeAMR's `interpolate` with `Lagrange(2)` — bilinear between cell
centres, which puts no value outside the range of its four neighbours, so a
contact stays a contact. Every point is folded onto the half box first; `ρ` is
even in the mirror, so the fold's sign does not enter.
"""
function sample_rect(Pfs, C, Wx, Wy, nx, ny, backend)
    n = nx * ny
    host, dev = pointbuffers(n, backend)
    x0, y0 = C[1] - Wx / 2, C[2] - Wy / 2
    Threads.@threads for j in 1:ny
        y = y0 + (j - 1 // 2) / ny * Wy
        for i in 1:nx
            x = x0 + (i - 1 // 2) / nx * Wx
            x′, y′, _ = fold(x, y)
            @inbounds host[i + (j - 1) * nx] = (x′, y′)
        end
    end
    dev === host || copyto!(dev, host)
    r = interpolate(Pfs, dev, Lagrange(2); vars=1:1)
    return reshape(Array(r.values), nx, ny)
end

# Box-average an `(s nx) × (s ny)` sample down to `nx × ny`.
function downsample(A::Matrix, s)
    s == 1 && return A
    nx, ny = size(A) .÷ s
    B = zeros(eltype(A), nx, ny)
    Threads.@threads for j in 1:ny
        for i in 1:nx
            acc = zero(eltype(A))
            for jj in 1:s, ii in 1:s
                acc += A[(i - 1) * s + ii, (j - 1) * s + jj]
            end
            B[i, j] = acc / s^2
        end
    end
    return B
end

"""
    find_target(Pfs, guess, mode, backend) -> point

The zoom target nearest `guess`, found in the state the zoom starts from
rather than in a pilot's: `"braid"` is the pressure maximum on the contact
`ρ ≈ (ρ₁ + ρ₂)/2` — the stagnation point of the braid between two billows,
where the sheet is stretched thinnest — and `"core"` the pressure minimum, a
billow's centre. Searched on a 256² grid over ±0.1 about the guess, in the
guess's own image of the reconstructed plane, so the anchor does not jump to
another periodic copy.
"""
function find_target(Pfs, guess, mode, backend; halfwidth=0.1, n=256)
    host, dev = pointbuffers(n * n, backend)
    xs = [guess[1] + ((i - 0.5) / n - 0.5) * 2halfwidth for i in 1:n]
    ys = [guess[2] + ((j - 0.5) / n - 0.5) * 2halfwidth for j in 1:n]
    for j in 1:n, i in 1:n
        x′, y′, _ = fold(xs[i], ys[j])
        host[i + (j - 1) * n] = (x′, y′)
    end
    dev === host || copyto!(dev, host)
    v = Array(interpolate(Pfs, dev, Lagrange(2); vars=[1, 4]).values)
    best, at = mode == "core" ? Inf : -Inf, guess
    for j in 1:n, i in 1:n
        ρ, p = v[1, 1, i + (j - 1) * n], v[2, 1, i + (j - 1) * n]
        if mode == "core"
            p < best && ((best, at) = (p, (xs[i], ys[j])))
        elseif abs(ρ - 3 // 2) < 1 // 10
            p > best && ((best, at) = (p, (xs[i], ys[j])))
        end
    end
    isfinite(best) || error("find_target: no $mode point within $halfwidth of $guess")
    return at
end

# The gas velocity at a reconstructed point: `v_y` takes the fold's sign,
# being the velocity normal to the mirror.
function velocity_at(Pfs, X, backend)
    x′, y′, s = fold(X[1], X[2])
    host = [(x′, y′)]
    dev = backend isa CPU ? host : (d = allocate(backend, NTuple{2,T}, 1); copyto!(d, host); d)
    v = Array(interpolate(Pfs, dev, Lagrange(2); vars=2:3).values)
    return (v[1, 1, 1], s * v[2, 1, 1])
end

"""
    view_blocks(forest, C, Wx, Wy, width_px, min_px) -> Matrix{Float32}

Every image of every leaf block that meets the view and is at least `min_px`
pixels wide, as columns `(level, u₀, v₀, u₁, v₁)` in view coordinates (0 to 1,
left to right and bottom to top): what the renderer draws the mesh from. A
block in the half box has an image in every tile of the reconstructed plane,
mirrored in the odd ones.
"""
function view_blocks(forest, C, Wx, Wy, width_px, min_px)
    x0, x1 = C[1] - Wx / 2, C[1] + Wx / 2
    y0, y1 = C[2] - Wy / 2, C[2] + Wy / 2
    out = Float32[]
    for k in forest.leaves
        e = block_extent(forest, k)
        (e[1][2] - e[1][1]) / Wx * width_px ≥ min_px || continue
        for m in floor(Int, 2y0):floor(Int, 2y1)
            b0, b1 = iseven(m) ? (e[2][1] + m / 2, e[2][2] + m / 2) :
                     ((m + 1) / 2 - e[2][2], (m + 1) / 2 - e[2][1])
            (b0 < y1 && b1 > y0) || continue
            for n in floor(Int, x0 - 1):floor(Int, x1)
                a0, a1 = e[1][1] + n, e[1][2] + n
                (a0 < x1 && a1 > x0) || continue
                append!(out, (level(k), (a0 - x0) / Wx, (b0 - y0) / Wy,
                              (a1 - x0) / Wx, (b1 - y0) / Wy))
            end
        end
    end
    return reshape(out, 5, :)
end

quantize(A, lo, hi) = map(a -> round(UInt16, clamp((a - lo) / (hi - lo), 0, 1) * 65535), A)

plainattr(x::Symbol) = String(x)
plainattr(x::Tuple) = collect(x)
plainattr(x) = x

function write_frame(dir, fr::Frame, ρ, mini, blocks, cfg, meta)
    lo, hi = Float64.(cfg["movie"]["rho_range"])
    path = joinpath(dir, "frames", "f$(lpad(fr.k, 6, '0')).h5")
    tmp = path * ".partial"
    h5open(tmp, "w") do f
        f["rho"] = quantize(ρ, lo, hi)
        f["minimap"] = quantize(mini, lo, hi)
        f["blocks"] = blocks
        a = attrs(f)
        a["rho_range"] = [lo, hi]
        for (k, v) in pairs(meta)
            a[string(k)] = plainattr(v)
        end
    end
    mv(tmp, path; force=true)
    return path
end

# --- the windowed criterion ------------------------------------------------------

"""
    windowed(flags, forest, cap) -> flags

`hydro_flags`' verdict, bounded per block by `cap(extent)`: a block above its
cap is told to coarsen whatever its indicator says, and a `Refine` at or above
it becomes a `Keep` that keeps its box — so it still recruits its neighbours
to its own level, which is what extends a window by one ring of blocks and
keeps it from flapping at its edge.
"""
function windowed(flags, forest, cap)
    return map(enumerate(flags)) do (b, f)
        k = forest.leaves[b]
        c = cap(block_extent(forest, k))
        verdict = f isa Tuple ? f[1] : f
        level(k) > c && return Coarsen
        verdict == Refine && level(k) ≥ c && return (Keep, f[2])
        return f
    end
end

# --- the run -------------------------------------------------------------------

function parse_args(args)
    opts = Dict{String,String}()
    for a in args
        m = match(r"^--([a-z-]+)=(.*)$", a)
        m === nothing && error("unrecognized argument $(repr(a)); see the header of simulate.jl")
        opts[m.captures[1]] = m.captures[2]
    end
    haskey(opts, "config") && haskey(opts, "out") || error("--config= and --out= are required")
    return opts
end

function main(args=ARGS)
    opts = parse_args(args)
    cfg = load_config(opts["config"])
    return withbackend(get(opts, "backend", "cpu"), T) do backend
        run_showcase(cfg, opts["out"], backend;
                     restart=get(opts, "restart", nothing),
                     stop_after=parse(Int, get(opts, "stop-after", "-1")),
                     walltime_hours=haskey(opts, "walltime-hours") ?
                                    parse(Float64, opts["walltime-hours"]) : nothing,
                     config_path=opts["config"])
    end
end

function run_showcase(cfg, out, backend; restart=nothing, stop_after=-1,
                      walltime_hours=nothing, config_path)
    wall0 = time()
    mkpath(joinpath(out, "frames"))
    mkpath(joinpath(out, "checkpoints"))
    cp(config_path, joinpath(out, "config.toml"); force=true)

    g = Geometry(cfg)
    sched = Schedule(cfg, g)
    frames = sched.frames
    ph, sc, mv_, cam, mesh = cfg["physics"], cfg["scheme"], cfg["movie"], cfg["camera"],
                             cfg["mesh"]
    N = Int(mesh["N"])
    cfl = T(ratio(sc["cfl"]))
    headroom = T(ratio(sc["headroom"]))
    refine_tol, coarsen_tol = ratio(sc["refine_tol"]), ratio(sc["coarsen_tol"])
    limiter, riemann = Symbol(sc["limiter"]), Symbol(sc["riemann"])
    ops = Operators(family=Conservative, prolongation=Int(sc["prolongation"]),
                    restriction=2)
    w = KelvinHelmholtz(T, Val(2); L=ratio(ph["L"]), a=ratio(ph["a"]), seed=:mirrored)
    case = HydroCase(w; roots=Int(mesh["roots"]), speed_headroom=headroom, half=true)
    eos, floors = case.eos, case.floors
    W, H, S = Int(mv_["width"]), Int(mv_["height"]), Int(mv_["supersample"])
    M = Int(mv_["minimap"])
    walltime = 3600 * something(walltime_hours, Float64(cfg["run"]["walltime_hours"]))

    write_schedule(joinpath(out, "schedule.tsv"), sched, g)
    @info "showcase: $(length(frames)) frames, $(sched.ksim + 1) simulated, " *
          "t_final = $(frames[sched.ksim + 1].t), levels $(g.ℓ_floor) → $(g.ℓ_max) " *
          "over $(sched.doublings) doublings, n_min = $(g.n_min), backend $backend"

    acc = ResetAccounting{T}(4; measure=true)
    # The camera's anchor and the gas velocity there: fixed at the target
    # through the intro, then advected with the gas (Heun at frame cadence).
    P = Tuple(Float64.(cam["target"]))
    vP = (0.0, 0.0)
    Ppred = P
    hits = ghosts = nsteps = nregrids = 0

    # How many chunks a frame interval is split into, so that the travel
    # margin stays within half a block at the view's level.
    nsub_for(fr, λ, Δt) = max(1, ceil(Int, headroom * λ * Δt /
                                           (spacing_at(fr.ℓ_view, g) * (N ÷ 2 - 1))))

    newproblem(U; prims=nothing, fluxes=nothing) =
        HydroProblem(U, ops; eos=eos, floors=floors, limiter=limiter, riemann=riemann,
                     fixup=true, boundary=nothing, prims=prims, fluxes=fluxes,
                     accounting=acc)

    if restart === nothing
        forest = Forest{T}(case.roots; N=N, periodic=case.periodic,
                           reflecting=case.reflecting, extents=case.extents)
        U = FieldSet{T}(forest, 4; G=2, parity=state_parity(forest, 4), backend=backend)
        initial_U = TreeHydro.conserved_initial(case)
        fill_by_coordinates!(initial_U, U)
        λ0 = max_signal_speed(TreeHydro.scratch_primitives(U, eos, floors))
        floorcap = _ -> g.ℓ_floor
        Δt1 = frames[2].t - frames[1].t
        criterion(fs) = windowed(hydro_flags(TreeHydro.scratch_primitives(fs, eos, floors);
                                             refine_tol=refine_tol, coarsen_tol=coarsen_tol,
                                             maxlevel_cap=g.ℓ_floor),
                                 fs.forest, floorcap)
        buffer0 = refinement_buffer(forest, g.ℓ_floor,
                                    headroom * λ0 * Δt1 / nsub_for(frames[2], λ0, Δt1))
        _, passes, converged = adapt_to_initial_data!(U, ops; initial=initial_U,
                                                      flags=criterion, buffer=buffer0,
                                                      maxpasses=g.ℓ_floor + 3)
        converged || error("the initial-data cycle did not converge in $passes passes")
        p = newproblem(U)
        u = statevector(U)
        gather!(u, U)
        k0 = 0
    else
        ck = load_checkpoint(restart; backend=backend)
        first(ck.application) == first(APPLICATION) ||
            error("$restart is not a showcase checkpoint")
        d = ck.data
        forest = ck.forest
        U, u = ck.fieldsets["U"].fieldset, ck.fieldsets["U"].state
        k0 = Int(d.frame)
        frames[k0 + 1].t == only(d.t) || error(
            "$restart ends at t = $(only(d.t)), and this configuration puts frame " *
            "$k0 at t = $(frames[k0 + 1].t): the schedule changed, which a restart " *
            "cannot continue (only the camera's tracking and the run keys may change)")
        P, vP, Ppred = Tuple(d.P), Tuple(d.vP), Tuple(d.Ppred)
        hits, ghosts, nsteps, nregrids = d.hits, d.ghosts, d.nsteps, d.nregrids
        acc.injection .= d.injection
        p = newproblem(U)
        @info "restarted from $restart at frame $k0, t = $(frames[k0 + 1].t)"
    end

    # One frame: sample, record, write. `P` must be current.
    function emit(fr::Frame, timings)
        tstart = time()
        C = camera_centre(fr, P, cfg, sched)
        Wx, Wy = view_size(fr, g)
        ρ = downsample(sample_rect(p.P, C, Wx, Wy, S * W, S * H, backend), S)
        mini = sample_rect(p.P, BOX_CENTRE, 1.0, 1.0, M, M, backend)
        blocks = view_blocks(forest, C, Wx, Wy, W, Float64(mv_["block_min_px"]))
        levels = zeros(Int, g.ℓ_max + 2)
        for k in forest.leaves
            levels[level(k) + 1] += 1
        end
        tsample = time() - tstart
        meta = (; k=fr.k, tau=fr.τ, t=fr.t, zeta=fr.ζ, zoom=2.0^fr.ζ, phase=fr.phase,
                centre=C, anchor=P, view=(Wx, Wy), level_view=fr.ℓ_view,
                level_finest=maxlevel(forest), nblocks=nleaves(forest),
                ncells=nleaves(forest) * N^2, levels=levels,
                h_min=minimum_spacing(T, forest), nsteps=nsteps, nregrids=nregrids,
                wall=time() - wall0, wall_step=timings.step, wall_regrid=timings.regrid,
                wall_sample=tsample, steps=timings.steps, subchunks=timings.nsub)
        write_frame(out, fr, ρ, mini, blocks, cfg, meta)
        open(joinpath(out, "log.tsv"), "a") do io
            println(io, join((fr.k, fr.phase, round(fr.t; sigdigits=10),
                              round(2.0^fr.ζ; sigdigits=6), fr.ℓ_view, maxlevel(forest),
                              nleaves(forest), nleaves(forest) * N^2, timings.steps,
                              timings.nsub, round(timings.step; digits=3),
                              round(timings.regrid; digits=3), round(tsample; digits=3),
                              round(time() - wall0; digits=1), round(Sys.maxrss() / 2^30; digits=2)),
                         '\t'))
        end
        return ρ
    end

    # The regrid for the chunk ending at frame `fr`, with the camera at `C`.
    function regrid_for(fr::Frame, C, λ, Δt)
        ℓv = fr.ℓ_view
        flags = windowed(hydro_flags(p; refine_tol=refine_tol, coarsen_tol=coarsen_tol,
                                     maxlevel_cap=ℓv),
                         forest, ext -> block_cap(ext, C, ℓv, g))
        bufferwidth = refinement_buffer(forest, ℓv, headroom * λ * Δt)
        pairs = (U => p.schedule, p.P => nothing, ntuple(d -> p.fluxes[d] => nothing, 2)...)
        if regrid!(forest, pairs; flags=flags, buffer=bufferwidth, boundary=nothing)
            nregrids += 1
            p = newproblem(U; prims=p.P, fluxes=p.fluxes)
            u = statevector(U)
            gather!(u, U)
            reset_atmosphere!(u, nothing, p, T(fr.t))
            return true
        end
        return false
    end

    function checkpoint(k, tag)
        path = joinpath(out, "checkpoints", "$(tag)_f$(lpad(k, 6, '0')).h5")
        save_checkpoint(path, forest; fieldsets=("U" => (U, u),), application=APPLICATION,
                        data=(; frame=k, t=[frames[k + 1].t], P=collect(P), vP=collect(vP),
                              Ppred=collect(Ppred), hits, ghosts, nsteps, nregrids,
                              injection=collect(acc.injection)),
                        sync=true)
        @info "checkpoint $path"
        return path
    end

    update_primitives!(p, u)
    k0 == 0 && emit(frames[1], (; step=0.0, regrid=0.0, steps=0, nsub=0))
    # A restart still in the intro takes the anchor from *this* configuration:
    # the target is where the zoom will go, and choosing it from the intro's
    # own frames is what a restart from the end of the intro is for. The mesh
    # of the intro and of the first doubling covers the whole box, so it does
    # not depend on the target.
    if k0 > 0 && frames[k0 + 1].phase === :intro
        P = Tuple(Float64.(cam["target"]))
        vP = cam["track"] === true ? velocity_at(p.P, P, backend) : (0.0, 0.0)
        Δt_next = frames[k0 + 2].t - frames[k0 + 1].t
        Ppred = (P[1] + Δt_next * vP[1], P[2] + Δt_next * vP[2])
    end
    integ_prev = nothing
    frame_max = 0.0
    # The regrid after a frame is the next frame's cost: it prepares the
    # mesh that frame is computed on, so it is logged there.
    tregrid_carry = 0.0
    tracking = cam["track"] === true
    lock_gain = Float64(cam["lock_gain"])
    target = Tuple(Float64.(cam["target"]))
    drift = Tuple(Float64.(cam["drift"]))
    t_target = Float64(cam["target_time"])

    for k in (k0 + 1):sched.ksim
        tframe = time()
        fr, prev = frames[k + 1], frames[k]
        Δt = fr.t - prev.t
        λ = max_signal_speed(p)
        nsub = nsub_for(fr, λ, Δt)
        tstep = 0.0
        tregrid = tregrid_carry
        steps_frame = 0
        for j in 1:nsub
            ta = T(prev.t + (j - 1) * Δt / nsub)
            tb = j == nsub ? T(fr.t) : T(prev.t + j * Δt / nsub)
            t1 = time()
            update_primitives!(p, u)
            λ = max_signal_speed(p)
            h_min = minimum_spacing(T, forest)
            dt = hydro_dt(forest, cfl, headroom * λ, Val(2))
            steps = max(1, ceil(Int, (tb - ta) / dt))
            integ = TreeHydro.hydro_integrator(p, u, ta, tb, steps; reset=:stage,
                                               alias_u0=true, reuse=integ_prev)
            IRK.solve!(integ)
            integ_prev = integ
            nsteps += steps
            steps_frame += steps
            update_primitives!(p, u)
            λ_end = max_signal_speed(p)
            check_cfl((tb - ta) / steps, h_min, 2, cfl, λ_end; chunk=k, λ=λ,
                      headroom=headroom)
            hits += floor_hits(p)
            ghosts += ghost_floor_hits(p)
            tstep += time() - t1
            if j < nsub
                t2 = time()
                C = camera_centre(fr, Ppred, cfg, sched)
                regrid_for(fr, C, λ_end, Δt / nsub) && (integ_prev = nothing)
                tregrid += time() - t2
            end
        end

        # The anchor: before `target_time` it moves with the prescribed drift
        # onto the target; after it, Heun's corrector with the velocity at the
        # predicted point now, then the optional lock toward the structure.
        active = fr.phase === :zoom || fr.phase === :hold
        if fr.t < t_target
            P = (target[1] + drift[1] * (fr.t - t_target),
                 target[2] + drift[2] * (fr.t - t_target))
        elseif active && tracking
            vpred = velocity_at(p.P, Ppred, backend)
            P = (P[1] + Δt / 2 * (vP[1] + vpred[1]), P[2] + Δt / 2 * (vP[2] + vpred[2]))
        end
        ρ = emit(fr, (; step=tstep, regrid=tregrid, steps=steps_frame, nsub))
        if active && lock_gain > 0
            P = lock_anchor(P, ρ, fr, cfg, sched, g, lock_gain)
        end

        k == sched.ksim && break
        nxt = frames[k + 2]
        # The zoom starts at the next frame: find its target in this state.
        if fr.phase === :intro && nxt.phase === :zoom && cam["target_mode"] != "fixed"
            P = find_target(p.P, P, String(cam["target_mode"]), backend)
            @info "zoom target ($(cam["target_mode"])) at t = $(fr.t): $P"
        end
        vP = nxt.t ≤ t_target ? drift :
             (active || nxt.phase === :zoom) && tracking ? velocity_at(p.P, P, backend) :
             (0.0, 0.0)
        Δt_next = nxt.t - fr.t
        Ppred = (P[1] + Δt_next * vP[1], P[2] + Δt_next * vP[2])
        t2 = time()
        C = camera_centre(nxt, Ppred, cfg, sched)
        λ = max_signal_speed(p)
        regrid_for(nxt, C, λ, Δt_next / nsub_for(nxt, λ, Δt_next)) && (integ_prev = nothing)
        update_primitives!(p, u)
        tregrid_carry = time() - t2

        frame_max = max(frame_max, time() - tframe)
        levelup = nxt.ℓ_view > fr.ℓ_view && cfg["run"]["checkpoint_levels"] === true
        stopping = k == stop_after ||
                   (walltime > 0 && time() - wall0 + 3 * frame_max > walltime)
        (levelup || stopping) && checkpoint(k, stopping ? "stop" : "level$(nxt.ℓ_view)")
        if stopping
            @info "stopped after frame $k at t = $(fr.t): restart with --restart"
            return nothing
        end
    end

    # The final state, then the zoom-out and the final hold from it.
    checkpoint(sched.ksim, "final")
    for k in (sched.ksim + 1):(length(frames) - 1)
        emit(frames[k + 1], (; step=0.0, regrid=0.0, steps=0, nsub=0))
    end
    injection = Tuple(acc.injection)
    @info "showcase done: $nsteps steps, $nregrids mesh changes, floor/reset/ghost " *
          "hits $hits/$(acc.hits)/$ghosts, injection $injection, " *
          "$(round((time() - wall0) / 60; digits=1)) min"
    (hits, acc.hits, ghosts) == (0, 0, 0) ||
        @warn "the floors fired, which they should never do on this case"
    return nothing
end

"""
    lock_anchor(P, ρ, fr, cfg, sched, g, gain) -> P

Pull the anchor a fraction `gain` of the way toward the `|∇ρ|²`-weighted
centroid of the frame within `lock_radius` view heights of the anchor's screen
position — a camera that stays on the sheet when the tracer drifts off it.
"""
function lock_anchor(P, ρ, fr, cfg, sched, g, gain)
    C = camera_centre(fr, P, cfg, sched)
    Wx, Wy = view_size(fr, g)
    nx, ny = size(ρ)
    r = Float64(cfg["camera"]["lock_radius"]) * Wy
    sx = sy = sw = 0.0
    for j in 2:(ny - 1), i in 2:(nx - 1)
        x = C[1] + ((i - 0.5) / nx - 0.5) * Wx
        y = C[2] + ((j - 0.5) / ny - 0.5) * Wy
        d2 = (x - P[1])^2 + (y - P[2])^2
        d2 < r^2 || continue
        gx = ρ[i + 1, j] - ρ[i - 1, j]
        gy = ρ[i, j + 1] - ρ[i, j - 1]
        wgt = (gx^2 + gy^2) * exp(-d2 / (r / 2)^2)
        sx += wgt * x
        sy += wgt * y
        sw += wgt
    end
    sw > 0 || return P
    target = (sx / sw, sy / sw)
    step = (gain * (target[1] - P[1]), gain * (target[2] - P[2]))
    # At most a hundredth of a view height per frame, so the camera glides.
    s = hypot(step...)
    s > 0.01 * Wy && (step = step .* (0.01 * Wy / s))
    return (P[1] + step[1], P[2] + step[2])
end

function write_schedule(path, sched, g)
    open(path, "w") do io
        println(io, "k\tphase\ttau\tt\tzeta\tzoom\tlevel_view")
        for fr in sched.frames
            println(io, join((fr.k, fr.phase, fr.τ, fr.t, fr.ζ, 2.0^fr.ζ, fr.ℓ_view), '\t'))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
