# The shared, pure pieces of the Kelvin–Helmholtz zoom showcase: the
# configuration, the movie schedule, the camera, the folding of the
# reconstructed box onto the simulated half box, and the window that decides
# which refinement levels may exist where. Included by `simulate.jl` and
# `render.jl`; depends on nothing but TOML and Base, so the renderer needs no
# mesh and no device.
#
# Coordinates. The simulation runs the **half box** `[0, 1] × [0, ½]`,
# periodic in `x` and reflecting at `y = 0` and `y = ½`. What the camera
# looks at is the **reconstructed plane**: the full periodic McNally box
# `[0, 1)²`, repeated in both directions, whose upper half is the lower half
# mirrored. Every camera position, anchor and view rectangle below is in the
# reconstructed plane; `fold` maps a point of it onto the half box, and the
# interval folds map a rectangle of it onto the pieces of the half box it
# covers. See `README.md` for why, and for the knobs.

using TOML

# --- configuration ---------------------------------------------------------

# Every key a configuration may set, with its default. A key not listed here
# is refused, so that a typo is an error rather than a silently ignored knob.
const SHOWCASE_DEFAULTS = Dict{String,Any}(
    "physics" => Dict{String,Any}(
        "L" => "1/40",          # McNally's ramp width
        "a" => "1/100",         # the seeded mode's amplitude
    ),
    "scheme" => Dict{String,Any}(
        "limiter" => "mc",
        "riemann" => "hllc",
        "cfl" => "2/5",
        "headroom" => "21/20",  # the deep-cap value, see CODE.md "Raising the level cap"
        "prolongation" => 3,
        "refine_tol" => "2/25",
        "coarsen_tol" => "1/50",
    ),
    "mesh" => Dict{String,Any}(
        "roots" => 2,           # root blocks along x; the half box has (roots, roots/2)
        "N" => 16,
        "n_min" => 128,         # finest cells across the view height, at least
        "window" => 1.25,       # margin factor of each level's box over its largest view
        "max_level" => 12,      # the finest refinement level the zoom reaches
    ),
    "movie" => Dict{String,Any}(
        "width" => 960,
        "height" => 540,
        "supersample" => 1,     # s×s samples per pixel, box-averaged
        "fps" => 30,
        "V0" => 0.15,           # simulated time per movie second at zoom 1
        "intro" => 10.0,        # movie seconds at zoom 1 before the zoom starts
        "T_level" => 5.0,       # movie seconds per zoom doubling
        "ease" => 2.0,          # movie seconds of ease-in and of ease-out
        "hold" => 4.0,          # movie seconds at the deepest zoom
        "T_out" => 0.75,        # movie seconds per doubling of the zoom-out
        "ease_out" => 1.5,
        "final" => 4.0,         # movie seconds at zoom 1 at the end, panning home
        "minimap" => 256,       # side of the full-box minimap, in samples
        "rho_range" => [0.9, 2.1],   # the UInt16 quantization range of ρ
        "block_min_px" => 3.0,  # block outlines smaller than this are not stored
    ),
    "camera" => Dict{String,Any}(
        "target" => [0.5, 0.25],     # the anchor at the zoom start, reconstructed plane
        "target_mode" => "fixed",    # or "braid"/"core": the nearest such point to `target`,
                                     # found in the run's own state when the zoom starts
        "target_time" => -1.0,       # when the anchor is at `target`; before the zoom
                                     # start (the default) means at the zoom start
        "drift" => [0.0, 0.0],       # the anchor's velocity before `target_time`: a
                                     # target picked at a later time is reached by
                                     # moving with it, and tracked with the gas after
        "glide" => 3.0,         # doublings over which the anchor glides to the centre
        "track" => true,        # advect the anchor with the gas
        "lock_gain" => 0.0,     # pull per frame toward the |∇ρ|² centroid near the anchor
        "lock_radius" => 0.15,  # in view heights
    ),
    "run" => Dict{String,Any}(
        "checkpoint_levels" => true,  # a checkpoint at every new finest level
        "walltime_hours" => 0.0,      # 0 is no limit
    ),
)

# `"p/q"` as an exact rational, anything else as itself: the physics goes to
# `KelvinHelmholtz`, which converts to the working type, and a rational keeps
# the numbers the package's own tests measured.
function ratio(x)
    x isa AbstractString || return x
    m = match(r"^\s*(-?\d+)\s*/\s*(\d+)\s*$", x)
    m === nothing && throw(ArgumentError("expected a number or \"p/q\", got $(repr(x))"))
    return parse(Int, m.captures[1]) // parse(Int, m.captures[2])
end

"""
    load_config(path) -> Dict

The configuration at `path`, merged over the defaults, with every unknown
section or key refused.
"""
function load_config(path::AbstractString)
    user = TOML.parsefile(path)
    cfg = deepcopy(SHOWCASE_DEFAULTS)
    for (section, entries) in user
        haskey(cfg, section) || throw(ArgumentError(
            "$path: unknown section [$section]; known: $(sort(collect(keys(cfg))))"))
        for (k, v) in entries
            haskey(cfg[section], k) || throw(ArgumentError(
                "$path: unknown key $section.$k; known: " *
                "$(sort(collect(keys(cfg[section]))))"))
            cfg[section][k] = v
        end
    end
    return cfg
end

# --- geometry ----------------------------------------------------------------

"""
    Geometry(cfg)

The numbers the schedule, the camera and the window share: `rN`, the finest
cells per unit length at level 0 (the half box's root blocks are `1/roots`
wide); `n_min`; the aspect ratio of the frame; the window factor; the floor
level `ℓ_floor`, which every point of the box may always reach, and the
deepest level `ℓ_max`.
"""
struct Geometry
    rN::Int
    n_min::Int
    aspect::Float64
    window::Float64
    ℓ_floor::Int
    ℓ_max::Int
end

function Geometry(cfg)
    m, mv = cfg["mesh"], cfg["movie"]
    roots, N, n_min = Int(m["roots"]), Int(m["N"]), Int(m["n_min"])
    iseven(roots) || throw(ArgumentError("mesh.roots must be even, got $roots"))
    rN = roots * N
    aspect = mv["width"] / mv["height"]
    ℓ_floor = level_for(1.0, rN, n_min)
    ℓ_max = Int(m["max_level"])
    ℓ_max > ℓ_floor || throw(ArgumentError(
        "mesh.max_level = $ℓ_max is not above the level the full-box view " *
        "already needs, $ℓ_floor: there would be nothing to zoom into"))
    return Geometry(rN, n_min, aspect, Float64(m["window"]), ℓ_floor, ℓ_max)
end

# The finest level a view of zoom `Z` needs so that its height holds more
# than `n_min` finest cells: `h_ℓ = 1/(rN 2^ℓ)` against `W_y = 1/Z`. The view
# then holds `(n_min, 2 n_min]` cells — a sawtooth, one level added at each
# doubling of the zoom.
level_for(Z, rN, n_min) = max(0, floor(Int, log2(Z * n_min / rN) + 1e-12) + 1)
level_for(Z, g::Geometry) = level_for(Z, g.rN, g.n_min)

# The largest zoom that level `ℓ` still suffices for: views of zoom in
# `[Z_ℓ(ℓ − 1), Z_ℓ(ℓ))` need level `ℓ`.
zoom_limit(ℓ, g::Geometry) = 2.0^ℓ * g.rN / g.n_min

spacing_at(ℓ, g::Geometry) = 1 / (g.rN * 2.0^ℓ)

# --- the movie schedule ------------------------------------------------------

smoothstep(u) = u ≤ 0 ? 0.0 : u ≥ 1 ? 1.0 : u * u * (3 - 2u)
# ∫₀ᵘ smoothstep, continued linearly past 1.
ismooth(u) = u ≤ 0 ? 0.0 : u ≥ 1 ? 0.5 + (u - 1) : u^3 - u^4 / 2

# `ζ = log₂ Z` along a ramp of `L` doublings at one doubling per `T` seconds,
# eased in and out over `E` seconds each: the rate rises smoothly from 0 to
# `1/T`, holds, and falls back to 0, so the zoom never jerks. Its duration is
# `L T + E`.
function ramp(s, L, T, E)
    dur = L * T + E
    s ≤ 0 && return 0.0
    s ≥ dur && return Float64(L)
    return (E * ismooth(s / E) - E * ismooth((s - (dur - E)) / E)) / T
end

"""
    Frame

One frame of the movie: its index `k` (from 0), movie time `τ`, simulated time
`t`, `ζ = log₂ Z`, the phase (`:intro`, `:zoom`, `:hold`, `:out`, `:final`),
and the finest level its view needs.
"""
struct Frame
    k::Int
    τ::Float64
    t::Float64
    ζ::Float64
    phase::Symbol
    ℓ_view::Int
end

"""
    Schedule(cfg, g) -> (; frames, ksim, τ)

Every frame of the movie, precomputed from the configuration alone — the zoom
is prescribed and so is the clock, so a restart recomputes exactly the same
times. `ksim` is the last frame the simulation advances to; every later frame
is taken from the final state. `τ` holds the five phase boundaries.

The clock is `dt/dτ = V₀ / Z`: slowing the simulation by the zoom factor keeps
a feature's speed in pixels constant, because an inviscid vortex sheet has no
length scale of its own — a disturbance of size `ℓ` on a sheet of jump `ΔU`
grows, and is carried across its own size, on a time of order `ℓ/ΔU`. See
`README.md`.
"""
function Schedule(cfg, g::Geometry)
    mv = cfg["movie"]
    fps = Int(mv["fps"])
    L = g.ℓ_max - g.ℓ_floor              # doublings from the floor to ℓ_max
    T, E = Float64(mv["T_level"]), Float64(mv["ease"])
    To, Eo = Float64(mv["T_out"]), Float64(mv["ease_out"])
    V0 = Float64(mv["V0"])
    τ1 = Float64(mv["intro"])
    τ2 = τ1 + L * T + E
    τ3 = τ2 + Float64(mv["hold"])
    τ4 = τ3 + L * To + Eo
    τ5 = τ4 + Float64(mv["final"])
    function ζ_at(τ)
        τ < τ1 && return 0.0
        τ < τ2 && return ramp(τ - τ1, L, T, E)
        τ < τ3 && return Float64(L)
        τ < τ4 && return L - ramp(τ - τ3, L, To, Eo)
        return 0.0
    end
    phase_at(τ) = τ < τ1 ? :intro : τ < τ2 ? :zoom : τ < τ3 ? :hold :
                  τ < τ4 ? :out : :final
    # The simulated time by composite Simpson over each frame interval — the
    # rate `V₀ 2^{−ζ}` is smooth there — frozen once the hold ends.
    rate(τ) = V0 * 2.0^(-ζ_at(τ))
    nframes = floor(Int, τ5 * fps) + 1
    frames = Frame[]
    t = 0.0
    ksim = 0
    for k in 0:(nframes - 1)
        τ = k / fps
        if k > 0 && τ ≤ τ3
            a, b, n = (k - 1) / fps, τ, 16
            h = (b - a) / n
            s = rate(a) + rate(b)
            for i in 1:(n - 1)
                s += (isodd(i) ? 4 : 2) * rate(a + i * h)
            end
            t += s * h / 3
            ksim = k
        end
        ζ = ζ_at(τ)
        # The level a frame needs, from its zoom; the frozen frames after the
        # hold read the final mesh and need nothing new.
        ℓ = min(level_for(2.0^ζ, g), g.ℓ_max)
        push!(frames, Frame(k, τ, t, ζ, phase_at(τ), ℓ))
    end
    return (; frames, ksim, τ=(τ1, τ2, τ3, τ4, τ5), doublings=L)
end

# --- folding the reconstructed plane onto the half box -----------------------

"""
    fold(x, y) -> (x′, y′, s)

The point of the half box `[0, 1) × [0, ½]` that the reconstructed point `(x,
y)` is an image of, and the sign `s` a velocity normal to the mirror (`v_y`)
takes there: periodic with period 1 in both directions, and mirrored in `y =
½`. Exact for the dyadic coordinates a mesh produces.
"""
@inline function fold(x, y)
    x′ = mod(x, one(x))
    y′ = mod(y, one(y))
    y′ > 1 // 2 && return (x′, one(y) - y′, -1)
    return (x′, y′, 1)
end

# Whether the reconstructed interval `[a, b]` meets `[lo, hi] ⊂ [0, 1]` after
# the periodic fold in `x`.
function xoverlap(a, b, lo, hi)
    b - a ≥ 1 && return true
    for n in floor(Int, a):floor(Int, b)
        p, q = max(a, n) - n, min(b, n + 1) - n
        p < hi && q > lo && return true
    end
    return false
end

# Whether the reconstructed interval `[a, b]` meets `[lo, hi] ⊂ [0, ½]` after
# the periodic-and-mirrored fold in `y`: tile `m` is `[m/2, (m+1)/2]`, a copy
# of the half box for even `m` and its mirror image for odd `m`.
function yoverlap(a, b, lo, hi)
    b - a ≥ 1 && return true
    for m in floor(Int, 2a):floor(Int, 2b)
        p, q = max(a, m / 2), min(b, (m + 1) / 2)
        p′, q′ = iseven(m) ? (p - m / 2, q - m / 2) : ((m + 1) / 2 - q, (m + 1) / 2 - p)
        p′ < hi && q′ > lo && return true
    end
    return false
end

# --- the camera ----------------------------------------------------------------

const BOX_CENTRE = (0.5, 0.5)

"""
    camera_centre(frame, P, cfg, sched) -> C

Where the camera looks in frame `frame`, given the anchor `P` (the tracked
point of the gas the zoom goes into, in the reconstructed plane):

  * the intro looks at the box;
  * the zoom and the hold put `P` where it is on the box view and glide it to
    the screen centre over the first `glide` doublings —
    `C = P + (C_box − P)·g(ζ)/Z` with `g` easing from 1 to 0 — so that zooming
    is first about `P` and then onto it;
  * the zoom-out stays centred on `P`, so that every view it shows lies inside
    the nested boxes the final mesh holds at full resolution;
  * the final hold pans from `P` back to the periodic image of the box centre
    nearest it.
"""
function camera_centre(fr::Frame, P, cfg, sched)
    fr.phase === :intro && return BOX_CENTRE
    if fr.phase === :zoom || fr.phase === :hold
        g = 1 - smoothstep(fr.ζ / Float64(cfg["camera"]["glide"]))
        Z = 2.0^fr.ζ
        return (P[1] + (BOX_CENTRE[1] - P[1]) * g / Z,
                P[2] + (BOX_CENTRE[2] - P[2]) * g / Z)
    end
    fr.phase === :out && return (P[1], P[2])
    # :final
    τ4, τ5 = sched.τ[4], sched.τ[5]
    λ = smoothstep((fr.τ - τ4) / max((τ5 - τ4) / 2, 1e-9))
    home = (BOX_CENTRE[1] + round(P[1] - BOX_CENTRE[1]), BOX_CENTRE[2])
    return (P[1] + λ * (home[1] - P[1]), P[2] + λ * (home[2] - P[2]))
end

view_size(fr::Frame, g::Geometry) = (g.aspect / 2.0^fr.ζ, 1 / 2.0^fr.ζ)

# --- the window ------------------------------------------------------------------

"""
    block_cap(ext, C, ℓ_view, g) -> Int

The finest level the block of physical extent `ext` (in the half box) may
have while the camera looks at `C` and the view needs level `ℓ_view`.

Level `ℓ` is allowed inside a box about `C` that covers, with the margin
factor `g.window`, the **largest view that needs level ℓ** — the view at zoom
`Z_ℓ(ℓ − 1)`, twice the height of the smallest. So a level, once present,
covers every view that will need it for the rest of the zoom, and the final
mesh covers every view of the zoom-out, which stays centred on the anchor.
Up to `ℓ_floor` every block may refine, and the box view of the intro and of
the final hold is at full resolution everywhere. Each box is twice the size of
the next, so each level holds about the same number of cells: `≈ 3·window²·
aspect·n_min²` if the criterion refined all of them, and the Löhner criterion
refines only what fires.
"""
function block_cap(ext, C, ℓ_view, g::Geometry)
    for ℓ in ℓ_view:-1:(g.ℓ_floor + 1)
        hy = g.window / (2 * zoom_limit(ℓ - 1, g))
        hx = g.aspect * hy
        xoverlap(C[1] - hx, C[1] + hx, ext[1][1], ext[1][2]) &&
            yoverlap(C[2] - hy, C[2] + hy, ext[2][1], ext[2][2]) && return ℓ
    end
    return g.ℓ_floor
end
