# The rendering half of the showcase: the frame files `simulate.jl` wrote,
# turned into a movie. Nothing here touches a mesh or a device — each frame is
# already the view, sampled — so this runs anywhere, and again with other
# colours or overlays without rerunning the simulation.
#
#     julia --project=showcase showcase/render.jl --frames=showcase/output/pilot \
#         --out=showcase/output/pilot/kh_zoom.mp4
#
# Options:
#   --frames=DIR     the simulation's output directory (required)
#   --out=FILE       .mp4 (H.264, yuv420p) or .gif; default DIR/kh_zoom.mp4
#   --colormap=NAME  any Makie colormap; default `lipari`
#   --mesh=MODE      block outlines: `zoomout` (fade in for the zoom-out, the
#                    default), `always` or `none`
#   --no-minimap, --no-text   leave out the inset, the captions
#   --title=TEXT     shown over the first seconds; `--title=` for none
#   --still=K        write frame K as a PNG next to --out instead of a movie
#   --from=K, --to=K a range of frames
#   --crf=N          the H.264 quality, lower is better; default 16
#
# Two Makie names collide with this environment's exports and are written in
# full: TreeAMR's `scatter!` and TreeHydro's `density` (see `CLAUDE.md`). This
# script loads neither package, so neither can bite, but the rule is kept.

using CairoMakie
using HDF5
using Printf

include(joinpath(@__DIR__, "kh_zoom.jl"))

function parse_render_args(args)
    opts = Dict{String,String}("colormap" => "lipari", "mesh" => "zoomout",
                               "title" => "Kelvin–Helmholtz instability",
                               "crf" => "16")
    for a in args
        if a in ("--no-minimap", "--no-text")
            opts[a[3:end]] = "yes"
            continue
        end
        m = match(r"^--([a-z-]+)=(.*)$", a)
        m === nothing && error("unrecognized argument $(repr(a)); see the header of render.jl")
        opts[m.captures[1]] = m.captures[2]
    end
    haskey(opts, "frames") || error("--frames= is required")
    get!(opts, "out", joinpath(opts["frames"], "kh_zoom.mp4"))
    return opts
end

framefiles(dir) = sort(filter(f -> occursin(r"^f\d{6}\.h5$", f), readdir(joinpath(dir, "frames"))))

struct FrameData
    ρ::Matrix{Float32}
    mini::Matrix{Float32}
    blocks::Matrix{Float32}
    meta::Dict{String,Any}
end

function readframe(path)
    h5open(path) do f
        a = attrs(f)
        meta = Dict{String,Any}(k => a[k] for k in keys(a))
        lo, hi = meta["rho_range"]
        dq(q) = Float32.(lo .+ (hi - lo) .* (read(q) ./ 65535))
        FrameData(dq(f["rho"]), dq(f["minimap"]), read(f["blocks"]), meta)
    end
end

# A length for the scale bar near a fifth of the view: 1, 2 or 5 times a
# power of ten, and its label.
function scalebar(Wx)
    target = Wx / 5
    e = floor(Int, log10(target))
    best = 10.0^e
    for m in (1, 2, 5, 10)
        m * 10.0^e ≤ target && (best = m * 10.0^e)
    end
    m = round(Int, best / 10.0^floor(Int, log10(best)))
    p = floor(Int, log10(best))
    label = p == 0 ? "$m" : m == 1 ? "10$(superscript(p))" : "$m×10$(superscript(p))"
    return best, label
end

superscript(n) = join(Dict('0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
                           '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹',
                           '-' => '⁻')[c] for c in string(n))

groupdigits(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => " ")

function zoomlabel(Z)
    Z < 10 && return @sprintf("zoom ×%.2f", Z)
    return "zoom ×" * groupdigits(round(Int, Z))
end

# The rectangle outline of each block, as pixel-space segments, and its
# colour by level.
function blocksegments(blocks, W, H, ℓmax, alpha)
    pts = Point2f[]
    cols = RGBAf[]
    cmap = cgrad(:turbo)
    for c in axes(blocks, 2)
        ℓ, u0, v0, u1, v1 = blocks[:, c]
        a, b, cc, d = u0 * W, v0 * H, u1 * W, v1 * H
        append!(pts, (Point2f(a, b), Point2f(cc, b), Point2f(cc, b), Point2f(cc, d),
                      Point2f(cc, d), Point2f(a, d), Point2f(a, d), Point2f(a, b)))
        col = cmap[clamp(ℓ / max(ℓmax, 1), 0, 1)]
        for _ in 1:8
            push!(cols, RGBAf(col.r, col.g, col.b, alpha))
        end
    end
    return pts, cols
end

function main(args=ARGS)
    opts = parse_render_args(args)
    dir = opts["frames"]
    cfg = load_config(joinpath(dir, "config.toml"))
    g = Geometry(cfg)
    sched = Schedule(cfg, g)
    files = framefiles(dir)
    isempty(files) && error("no frame files in $(joinpath(dir, "frames"))")
    # Only the frames the schedule has: a run restarted with a shorter zoom
    # leaves the longer run's later frames behind, which are not this movie.
    ks = [parse(Int, f[2:7]) for f in files]
    keep = ks .< length(sched.frames)
    ks, files = ks[keep], files[keep]
    from = parse(Int, get(opts, "from", string(first(ks))))
    to = parse(Int, get(opts, "to", string(last(ks))))
    if haskey(opts, "still")
        from = to = parse(Int, opts["still"])
    end
    sel = [(k, joinpath(dir, "frames", f)) for (k, f) in zip(ks, files) if from ≤ k ≤ to]
    isempty(sel) && error("no frames in $from:$to")
    # A missing frame would be a jump in the movie, so it is refused.
    [k for (k, _) in sel] == collect(first(sel)[1]:last(sel)[1]) ||
        error("frames missing between $from and $to: the simulation has not finished them")

    first_frame = readframe(last(sel[1]))
    W, H = size(first_frame.ρ)
    fps = Int(cfg["movie"]["fps"])
    lo, hi = 1.0f0, 2.0f0              # ρ₁ and ρ₂: the colour range is the physics
    cmap = Symbol(opts["colormap"])
    showmini = !haskey(opts, "no-minimap")
    showtext = !haskey(opts, "no-text")
    meshmode = opts["mesh"]
    title = opts["title"]
    ℓmax = g.ℓ_max

    CairoMakie.activate!(type="png", px_per_unit=1)
    scene = Scene(size=(W, H), camera=campixel!, backgroundcolor=:black)
    img = Observable(first_frame.ρ)
    image!(scene, 0 .. W, 0 .. H, img; colormap=cmap, colorrange=(lo, hi),
           interpolate=false)

    segs = Observable(Point2f[])
    segcols = Observable(RGBAf[])
    linesegments!(scene, segs; color=segcols, linewidth=1)

    # The minimap: the full reconstructed box at the frame's time, with the
    # view on it — a rectangle while it is large enough to see, a ring after.
    m = round(Int, 0.24 * H)
    mx0, my0 = W - m - round(Int, 0.03 * H), H - m - round(Int, 0.03 * H)
    mini = Observable(first_frame.mini)
    rect = Observable(Point2f[])
    ring = Observable(Point2f[Point2f(NaN, NaN)])
    if showmini
        image!(scene, mx0 .. (mx0 + m), my0 .. (my0 + m), mini; colormap=cmap,
               colorrange=(lo, hi), interpolate=false)
        lines!(scene, Point2f[(mx0, my0), (mx0 + m, my0), (mx0 + m, my0 + m),
                              (mx0, my0 + m), (mx0, my0)]; color=(:white, 0.8),
               linewidth=1.5)
        lines!(scene, rect; color=:white, linewidth=2)
        CairoMakie.scatter!(scene, ring; marker=:circle, markersize=round(Int, 0.03 * H),
                            color=:transparent, strokecolor=:white, strokewidth=2)
    end

    fs = round(Int, 0.028 * H)
    pad = round(Int, 0.03 * H)
    ttext = Observable("")
    ztext = Observable("")
    ltext = Observable("")
    barpts = Observable(Point2f[])
    bartext = Observable("")
    titlealpha = Observable(0.0)
    if showtext
        shadow = (:black, 0.6)
        for (obs, pos, align) in ((ttext, (pad, H - pad), (:left, :top)),
                                  (ztext, (pad, pad + 1.4fs), (:left, :bottom)),
                                  (ltext, (pad, pad), (:left, :bottom)))
            text!(scene, Point2f(pos...) .+ Point2f(1.5, -1.5); text=obs, align=align,
                  fontsize=fs, color=shadow)
            text!(scene, Point2f(pos...); text=obs, align=align, fontsize=fs, color=:white)
        end
        lines!(scene, barpts; color=:white, linewidth=3)
        text!(scene, lift(p -> isempty(p) ? Point2f(0, 0) : (p[1] + p[2]) / 2 .+ Point2f(0, 8),
                          barpts); text=bartext, align=(:center, :bottom), fontsize=fs,
              color=:white)
        if !isempty(title)
            text!(scene, Point2f(W / 2, H / 2); text=title, align=(:center, :center),
                  fontsize=round(Int, 0.07 * H), font=:bold,
                  color=lift(a -> (:white, a), titlealpha))
        end
    end

    τ4 = sched.τ[4]
    function update!(fd::FrameData)
        img[] = fd.ρ
        mt = fd.meta
        phase = Symbol(mt["phase"])
        τ = mt["tau"]
        Z = mt["zoom"]
        Wx, Wy = mt["view"]
        C = mt["centre"]
        # The mesh, faded in over the first second of the zoom-out.
        α = meshmode == "always" ? 0.5 :
            meshmode == "zoomout" && phase in (:out, :final) ?
            0.5 * smoothstep((τ - sched.τ[3]) / 1.0) : 0.0
        # Both inputs of the plot are set before it is drawn, which is when
        # Makie resolves them, so the two lengths never disagree there.
        if α > 0 && size(fd.blocks, 2) > 0
            p, c = blocksegments(fd.blocks, W, H, ℓmax, α)
            segcols[] = c
            segs[] = p
        else
            segcols[] = RGBAf[]
            segs[] = Point2f[]
        end
        if showmini
            mini[] = fd.mini
            # The view on the box, at the periodic image whose centre is in it.
            cx, cy = mod(C[1], 1.0), clamp(C[2], 0.0, 1.0)
            wx, wy = min(Wx, 1.0), min(Wy, 1.0)
            if wx * m ≥ 6
                x0 = clamp(cx - wx / 2, 0, 1 - wx)
                y0 = clamp(cy - wy / 2, 0, 1 - wy)
                rect[] = Point2f[(mx0 + x0 * m, my0 + y0 * m),
                                 (mx0 + (x0 + wx) * m, my0 + y0 * m),
                                 (mx0 + (x0 + wx) * m, my0 + (y0 + wy) * m),
                                 (mx0 + x0 * m, my0 + (y0 + wy) * m),
                                 (mx0 + x0 * m, my0 + y0 * m)]
                ring[] = [Point2f(NaN, NaN)]
            else
                rect[] = Point2f[]
                ring[] = [Point2f(mx0 + cx * m, my0 + cy * m)]
            end
        end
        if showtext
            ttext[] = @sprintf("t = %.6f", mt["t"])
            ztext[] = zoomlabel(Z)
            ℓ = Int(mt["level_finest"])
            ltext[] = "refinement level $ℓ  ·  finest cell 1/" *
                      groupdigits(g.rN * 2^ℓ) * " of the box"
            len, lab = scalebar(Wx)
            px = len / Wx * W
            x1 = W - pad
            barpts[] = Point2f[(x1 - px, pad + fs / 2), (x1, pad + fs / 2)]
            bartext[] = lab
            titlealpha[] = 1 - smoothstep((τ - 2.0) / 1.5)
        end
        return nothing
    end

    out = opts["out"]
    if haskey(opts, "still")
        update!(first_frame)
        path = replace(out, r"\.[a-z0-9]+$" => "") * "_f$(lpad(from, 6, '0')).png"
        save(path, scene)
        println(path)
        return path
    end
    t0 = time()
    n = length(sel)
    record(scene, out, eachindex(sel); framerate=fps, compression=parse(Int, opts["crf"]),
           profile="high", pixel_format="yuv420p") do i
        update!(readframe(last(sel[i])))
        i % 100 == 0 && @info "rendered $i of $n frames, $(round(time() - t0; digits=1)) s"
    end
    @info "wrote $out: $n frames at $fps fps in $(round(time() - t0; digits=1)) s"
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
