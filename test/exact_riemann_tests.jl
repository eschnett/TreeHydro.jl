# The exact Riemann solver: the first thing in this package whose answer is
# checked against somebody else's numbers.
#
# Everything measured so far has been measured against this code's own
# reference — an exact cell average of a sine, an analytic time derivative
# of the same. Those catch a scheme that is wrong; they cannot catch a
# *reference* that is wrong, because they are the reference. Toro's Table
# 4.3 is the outside check: five standard Riemann problems with their star
# regions tabulated, transcribed here and compared digit by digit.
#
# The three sets of values below are a transcription. Where the solver and
# the table disagree by more than the table's last printed digit, the right
# response is to report it, not to widen the tolerance — so the tolerance
# for each entry is *one unit in the last decimal the table prints*, which
# is the sharpest claim a transcription supports.
#
# Beyond the table, four structural claims that need no external numbers:
# equal states give a constant solution; mirroring the problem mirrors the
# solution; the sample is continuous at a fan edge and jumps at a shock; and
# the supremum of the signal speed exceeds anything the initial data
# carries, which is the whole reason `sod.jl` takes its `λ` from here.

# One Riemann problem: γ, the two states, and the tabulated star region with
# the number of decimals Toro prints for each entry.
struct ToroCase
    name::String
    γ::Float64
    left::NTuple{3,Float64}
    right::NTuple{3,Float64}
    star::NTuple{4,Float64}          # p★, u★, ρ★_L, ρ★_R
    decimals::NTuple{4,Int}
end

const TORO_TESTS = (
    ToroCase("1, Sod", 1.4, (1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
             (0.30313, 0.92745, 0.42632, 0.26557), (5, 5, 5, 5)),
    ToroCase("2, the 123 problem", 1.4, (1.0, -2.0, 0.4), (1.0, 2.0, 0.4),
             (0.00189, 0.0, 0.02185, 0.02185), (5, 5, 5, 5)),
    ToroCase("3, the left blast", 1.4, (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01),
             (460.894, 19.5975, 0.57506, 5.99924), (3, 4, 5, 5)),
)

toro_solve(c::ToroCase) = exact_riemann(c.γ, c.left..., c.right...)

@testset "The star region is Toro's: test $(c.name)" for c in TORO_TESTS
    # The claim that makes every Sod number downstream mean something. A
    # sign error in the shock branch of the pressure function, a
    # rarefaction exponent written `(γ−1)/2γ` where it should be its
    # reciprocal, a Newton step that converged to the wrong root — none of
    # them would show as anything but a slightly different reference, and a
    # slightly different reference is indistinguishable from a slightly
    # wrong scheme.
    #
    # The three problems are chosen for their three shapes: Sod is a fan
    # and a shock, the 123 problem is two strong rarefactions that nearly
    # produce a vacuum (the branch where the pressure iteration is most
    # fragile), and the left blast is a pressure ratio of 10⁵, where a
    # careless initial guess costs iterations or diverges.
    #
    # The tolerance is one unit in the last decimal the table prints. Every
    # entry also agrees to 1e-4 *relative* except the 123 problem's `p★`,
    # whose tabulated 0.00189 carries three significant digits: the solver
    # returns 0.00189387, which is 3.9e-6 absolute and 2.0e-3 relative — the
    # table's rounding, not the solver's error.
    sol = toro_solve(c)
    got = (sol.p★, sol.u★, sol.ρ★_L, sol.ρ★_R)
    for (name, g, want, dec) in zip(("p★", "u★", "ρ★_L", "ρ★_R"), got, c.star,
                                    c.decimals)
        @test abs(g - want) ≤ 10.0^(-dec)
        @info "Toro test $(c.name): $name = $g, table $want"
    end
    # Convergence is part of the claim: the pressure function is monotone
    # and convex, so a solver needing tens of steps on these has a guess
    # that is not Toro's even if the answer comes out.
    @test sol.iterations ≤ 8
end

@testset "Sod's waves are a fan, a contact and a shock" begin
    # The shape of the solution, not its numbers: Sod's left wave is a
    # rarefaction and its right wave a shock, and everything `sod.jl` does
    # with the speeds assumes the constructor got that right. A fan is
    # stored with a head and a tail that differ and a shock with both at one
    # speed, which is what lets `sample` ask the same two questions for
    # either; if a shock were stored as a fan, the sampler would interpolate
    # across it and the reference would have no discontinuity in it at all.
    sol = toro_solve(TORO_TESTS[1])
    @test !sol.shock_L
    @test sol.shock_R
    @test sol.head_L < sol.tail_L < sol.u★ < sol.tail_R
    @test sol.tail_R == sol.head_R          # a shock has one speed
    # The five speeds, for the record, and the numbers the arrival check
    # and the figures of step 11 will use.
    @info "Sod: fan head $(sol.head_L), fan tail $(sol.tail_L), contact " *
          "$(sol.u★), shock $(sol.head_R)"
end

@testset "Equal states give a constant solution" begin
    # The consistency property, and the one place the answer is exactly
    # representable: with both states equal the pressure function is zero at
    # `p★ = p`, the two star densities are the state's own, and every
    # region of the sample is the same state. Asserted bit for bit rather
    # than to a tolerance — the iteration's first step is a division of an
    # exact zero, so nothing rounds. A solver that drifted here would be
    # adding a wave where the physics has none.
    ρ, u, p = 1.3, 0.7, 2.1
    sol = exact_riemann(7 / 5, ρ, u, p, ρ, u, p)
    @test sol.p★ == p
    @test sol.u★ == u
    @test sol.ρ★_L == ρ
    @test sol.ρ★_R == ρ
    for ξ in (-Inf, -10.0, -1.0, 0.0, 0.7, 5.0, Inf)
        @test sample(sol, ξ) == (ρ, u, p)
    end
end

@testset "Mirroring the problem mirrors the solution" begin
    # `x → −x`, `u → −u` is an exact symmetry of the Euler equations, so the
    # solver must carry it: the mirrored problem has the same star pressure,
    # the negated star velocity, the two star densities exchanged, and each
    # side's wave kind exchanged with the other's. It catches an asymmetry
    # no tabulated number would — the left and right branches of the
    # pressure function and of the sampler are written out separately, and a
    # sign carried in one and not the other survives Sod (whose left state
    # is at rest) untouched.
    #
    # Not bit for bit: the two Newton iterations start from different
    # guesses and take different paths to the same root.
    γ = 7 / 5
    ρ_L, u_L, p_L = 1.0, 0.3, 1.0
    ρ_R, u_R, p_R = 0.2, -0.4, 0.15
    sol = exact_riemann(γ, ρ_L, u_L, p_L, ρ_R, u_R, p_R)
    mir = exact_riemann(γ, ρ_R, -u_R, p_R, ρ_L, -u_L, p_L)
    @test mir.p★ ≈ sol.p★ rtol = 1e-12
    @test mir.u★ ≈ -sol.u★ rtol = 1e-12
    @test mir.ρ★_L ≈ sol.ρ★_R rtol = 1e-12
    @test mir.ρ★_R ≈ sol.ρ★_L rtol = 1e-12
    @test mir.shock_L == sol.shock_R
    @test mir.shock_R == sol.shock_L
    for ξ in (-3.0, -1.2, -0.5, 0.0, 0.4, 1.1, 2.5)
        ρ, u, p = sample(sol, ξ)
        ρm, um, pm = sample(mir, -ξ)
        @test ρm ≈ ρ rtol = 1e-12
        @test um ≈ -u atol = 1e-12
        @test pm ≈ p rtol = 1e-12
    end
end

@testset "The sample is continuous at a fan edge and jumps at a shock" begin
    # What makes the reference a reference: the exact solution is smooth
    # through the head and the tail of a rarefaction and discontinuous
    # across the shock and the contact. A sampler whose fan formula did not
    # match its star state at the tail — the classic off-by-a-factor in
    # `2/(γ+1)` — would leave a small jump at a place the scheme is meant to
    # resolve smoothly, and the L1 error would stop converging at a rate
    # nobody could explain.
    sol = toro_solve(TORO_TESTS[1])
    ε = 1e-7
    for ξ in (sol.head_L, sol.tail_L)
        lo, hi = sample(sol, ξ - ε), sample(sol, ξ + ε)
        for v in 1:3
            @test lo[v] ≈ hi[v] atol = 1e-6
        end
    end
    # The shock: the density jumps by more than a factor of two, and the
    # contact separates two densities at one pressure and one velocity.
    pre, post = sample(sol, sol.head_R + ε), sample(sol, sol.head_R - ε)
    @test post[1] / pre[1] > 2
    @test post[3] / pre[3] > 2
    left, right = sample(sol, sol.u★ - ε), sample(sol, sol.u★ + ε)
    @test left[1] / right[1] > 3 / 2            # ρ jumps across the contact
    @test left[2] ≈ right[2] rtol = 1e-12       # u and p do not
    @test left[3] ≈ right[3] rtol = 1e-12
    # And the two ends are the initial states exactly, which is what makes
    # `t = 0` a legal argument to `sod_reference`.
    @test sample(sol, -Inf) == (sol.ρ_L, sol.u_L, sol.p_L)
    @test sample(sol, Inf) == (sol.ρ_R, sol.u_R, sol.p_R)
end

@testset "The fastest signal is not in the initial data" begin
    # The observation this step records, in its cheapest form. Sod's initial
    # data carries nothing faster than `c_L = sqrt(7/5) ≈ 1.183`; the state
    # behind the shock carries `u★ + c★_R ≈ 2.192`. A driver that sized its
    # step from the state it had at `t = 0` would run at 1.85 times the CFL
    # number it asked for, which is why `sod_errors` takes its `λ` from the
    # exact solution and why the chunked driver of step 7 needs a headroom
    # factor. See "Time integration and the time step" in `CODE.md`.
    sol = toro_solve(TORO_TESTS[1])
    λ = max_signal_speed(sol)
    λ_initial = max(abs(sol.u_L) + sol.c_L, abs(sol.u_R) + sol.c_R)
    @info "Sod: λ = $λ against λ from the initial data $λ_initial, " *
          "ratio $(λ / λ_initial)"
    @test λ ≈ 2.1916 atol = 1e-4
    @test λ > sol.c_L
    @test λ / λ_initial ≈ 1.8522 atol = 1e-4
    # It is the *right* star state that is fastest, which is the detail
    # worth writing down: the post-shock gas is the hotter of the two, so
    # `u★ + c★_R` beats the `u★ + c★_L ≈ 1.925` of the gas behind the
    # contact. Both exceed anything at `t = 0`.
    @test sol.u★ + sol.c★_R == λ
    @test sol.u★ + sol.c★_L ≈ 1.9252 atol = 1e-4
    # And a fan carries nothing its edges do not: `|u| + c` is convex and
    # piecewise linear in ξ there, so sampling the left fan finds no speed
    # above λ. Sampled densely because this is the one claim in the file
    # that is about an inequality holding everywhere rather than at a point.
    for ξ in range(sol.head_L, sol.tail_L; length=101)
        ρ, u, p = sample(sol, ξ)
        @test abs(u) + sqrt(sol.γ * p / ρ) ≤ λ
    end
end

@testset "The solver refuses the states it cannot solve" begin
    # Each of these would otherwise be found as a wandering Newton
    # iteration or a `NaN` in the reference. The vacuum case is the real
    # one: where the two gases separate faster than either can expand, the
    # exact solution has a vacuum region between two rarefactions, `p★` is
    # zero, and the pressure function has no positive root at all — so the
    # iteration would not fail, it would merely stop somewhere. The message
    # names the pressure positivity condition and both of its sides.
    @test_throws "generate a vacuum" exact_riemann(7 / 5, 1.0, -5.0, 0.4,
                                                   1.0, 5.0, 0.4)
    @test_throws "both states positive" exact_riemann(7 / 5, 1.0, 0.0, 1.0,
                                                      -1.0, 0.0, 0.1)
    @test_throws "both states positive" exact_riemann(7 / 5, 1.0, 0.0, 0.0,
                                                      1.0, 0.0, 0.1)
    @test_throws "needs γ > 1" exact_riemann(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.1)
    # And the one it must accept, right up against the condition: the 123
    # problem has `u_R − u_L = 4` against a bound of `2(c_L + c_R)/(γ−1) ≈
    # 7.48`, and doubling its velocities is what crosses it.
    @test exact_riemann(7 / 5, 1.0, -2.0, 0.4, 1.0, 2.0, 0.4).p★ > 0
end
