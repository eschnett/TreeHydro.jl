# **Long tier.** This file runs only under `TREEHYDRO_TEST_LONG=1`; see
# "Testing: two tiers" in `CODE.md`. It is where the physics is claimed and
# where the numbers `CODE.md` records come from, and it may take minutes.
# The short tier pins reduced configurations of the same studies against
# committed references in `test/regression_tests.jl`.
#
# The calibration: the measurement that chose the refinement tolerances.
#
# The indicator's unit claims — that τ fires at Sod's discontinuity and
# nowhere else, that ρ catches what p cannot, that the atmosphere does not
# fire and does without the global term, that a box bounds the firing cells,
# that the four marks are the four cases, and that the buffer refuses more
# than a block width — are in the short tier, in `test/refinement_tests.jl`,
# which also defines the helpers below. What is here is the part that costs
# eight evolutions: max τ against h at four spacings, for Sod's four
# features and for the McNally ramp. The short tier pins the `h = 1/64` row
# of both tables against a stored reference instead.

@testset "The calibration: max τ against h on uniform meshes" begin
    # Löhner's canonical `τ > 0.8` is a shock detector on *sharp* data, and
    # the thresholds this package needs are the ones a smooth ramp crosses,
    # so they had to be measured. Two tables, both on uniform meshes at
    # four spacings: Sod after a short evolution, read in a window around
    # each feature located from the exact solution; and the McNally density
    # ramp, which is the feature the criterion is meant to resolve *and
    # stop*. The numbers and the tolerances they chose are recorded under
    # "Step 6" in `CODE.md`'s "Measured results"; what is asserted here is
    # the shape of each column, which is what the choice rests on.
    T = Float64
    t_end = T(1 // 10)
    w = SodTube(T, Val(1))
    sol = exact_riemann(w)
    x₀ = w.x₀
    head = x₀ + sol.head_L * t_end
    tail = x₀ + sol.tail_L * t_end
    contact = x₀ + sol.u★ * t_end
    shock = x₀ + sol.head_R * t_end
    # Wide enough to hold three cells at the coarsest spacing and narrow
    # enough that the four windows stay disjoint.
    win = T(3 // 100)
    # The middle of the fan, away from both of its kinks.
    fan = (head + (tail - head) * 3 // 10, head + (tail - head) * 7 // 10)

    function maxwin(table, lo, hi)
        return maximum((c.τ for c in table if lo <= c.x[1] <= hi); init=zero(T))
    end

    Ns = (16, 32, 64, 128)
    sodτ = map(Ns) do N
        r = primed_sod(Val(1), N; roots=4, t_end=t_end, limiter=:minmod)
        table = tau_table(r.p)
        (h=minimum_spacing(T, r.forest),
         head=maxwin(table, head - win, head + win),
         tail=maxwin(table, tail - win, tail + win),
         contact=maxwin(table, contact - win, contact + win),
         shock=maxwin(table, shock - win, shock + win),
         fan=maxwin(table, fan[1], fan[2]))
    end
    for c in sodτ
        @info "Sod at t = 0.1, :minmod, HLLE, h = $(rationalize(c.h)): max τ " *
              "at the shock $(round(c.shock, digits=4)), the contact " *
              "$(round(c.contact, digits=4)), the fan's head " *
              "$(round(c.head, digits=4)) and tail $(round(c.tail, digits=4)), " *
              "inside the fan $(round(c.fan, digits=5))"
    end

    # The shock is the one feature whose τ does not fall: a captured shock
    # is the same few cells wide at every spacing, so the criterion never
    # resolves it and `maxlevel_cap` is what stops refinement there.
    @test all(c -> 0.4 < c.shock < 0.8, sodτ)
    @test maximum(c.shock for c in sodτ) / minimum(c.shock for c in sodτ) < 1.2
    # It stays far above any threshold a smooth feature would set.
    @test minimum(c.shock for c in sodτ) > 5 * T(REFINE_TOL)
    # The contact is nearly as persistent, and falls only as the HLL
    # diffusion spreads it over more cells.
    @test all(c -> c.contact > T(REFINE_TOL), sodτ)
    # The smooth interior of the fan falls with h, and fast: once the local
    # floor dominates the denominator the decay is second order.
    @test all(i -> sodτ[i].fan > sodτ[i + 1].fan, 1:(length(sodτ) - 1))
    @test sodτ[1].fan / sodτ[end].fan > 10
    # So do the fan's kinks — a numerically computed rarefaction head is
    # not a kink at all, the scheme having rounded it over a nearly fixed
    # physical width.
    @test all(i -> sodτ[i].head > sodτ[i + 1].head, 1:(length(sodτ) - 1))
    @test sodτ[1].tail > sodτ[end].tail

    # The McNally density ramp: ρ from 1 to 2 over an exponential ramp of
    # width L = 1/40 at uniform pressure, the shape of the Kelvin–Helmholtz
    # shear layer's density (see "Kelvin–Helmholtz instability" in
    # `CODE.md`; step 10 checks the setup against the paper, and only the
    # ramp's *shape* is used here). One dimension, no evolution: this is a
    # statement about initial data and the mesh, not about the scheme.
    ρ₁, ρ₂, Lr = one(T), T(2), T(1 // 40)
    ρm = (ρ₁ - ρ₂) / 2
    function ramp(x)
        y = x[1]
        ρ = y < T(1 // 4) ? ρ₁ - ρm * exp((y - T(1 // 4)) / Lr) :
            y < T(1 // 2) ? ρ₂ + ρm * exp((T(1 // 4) - y) / Lr) :
            y < T(3 // 4) ? ρ₂ + ρm * exp((y - T(3 // 4)) / Lr) :
            ρ₁ - ρm * exp((T(3 // 4) - y) / Lr)
        return (ρ, zero(T), T(5 // 2))
    end
    eos = IdealGas(T(5 // 3))
    rampτ = map(Ns) do N
        forest = Forest((4,); N=N, periodic=(true,),
                        extents=((zero(T), one(T)),))
        r = primed_problem(Val(1), forest, ramp; eos=eos, floors=quiet_floors())
        (h=minimum_spacing(T, forest), τ=maximum(c.τ for c in tau_table(r.p)))
    end
    for c in rampτ
        @info "McNally density ramp, L = 1/40, h = $(rationalize(c.h)): max τ " *
              "$(round(c.τ, digits=5))"
    end

    # It falls, and by more than a factor of two per halving — which is
    # what makes refinement of the shear layer terminate, unlike the shock.
    @test all(i -> rampτ[i].τ > 2 * rampτ[i + 1].τ, 1:(length(rampτ) - 1))
    # And `refine_tol` sits mid-plateau of the depth the ramp then reaches:
    # from a 1/64 base it refines twice, to 1/256, with room on both sides.
    @test rampτ[1].τ > T(REFINE_TOL)
    @test rampτ[2].τ > T(REFINE_TOL) > rampτ[3].τ
    @test rampτ[2].τ / T(REFINE_TOL) > 1.4
    @test T(REFINE_TOL) / rampτ[3].τ > 1.4
end
