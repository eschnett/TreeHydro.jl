# The time integrator: IMEXRungeKutta's `SSPRK33`, its stage arithmetic run
# by block owner, with the atmosphere reset in its two limiter hooks
# (amended after step 11, replacing OrdinaryDiffEq's `SSPRK33`; see "Time
# integration and the time step" in `CODE.md`).
#
# Why a second package and not OrdinaryDiffEq: OrdinaryDiffEq forms its
# stage vectors with a serial broadcast on one core, which is the Amdahl term
# TreeWave measured capping a threaded step at 3.6×, and it copies and
# allocates its buffers serially at every `solve`. IMEXRungeKutta forms each
# stage combination as one pass over the state, split by the partition below
# so that every block's entries are combined on the thread `map_blocks!` runs
# that block on, first-touches its scratch arrays through the same
# partition, and can take over the previous chunk's scratch. The result is
# bitwise the same on every path and at every thread count. It is the
# integrator TreeAMR's own tests and TreeGeneralizedHarmonic use.
#
# Its `SSPRK33` is the Shu–Osher (3,3) method written in **Butcher form**,
# and that moves where a limiter's correction goes: the stage limiter acts on
# a stage value just before the right-hand side reads it, and its correction
# reaches `uⁿ⁺¹` only *through* that right-hand side — which is a flux
# divergence and conserves — while only the step limiter writes `uⁿ⁺¹`
# itself. So only the step limiter moves a conserved total, and
# `reset_stage!` accounts no injection. See "Floors and the atmosphere" in
# `CODE.md`.

"""
    state_partition(U::FieldSet, u) -> Vector{UnitRange{Int}} or nothing

The block-ownership partition of the state vector `u` of `U`, as
IMEXRungeKutta's `partition` keyword takes it: element `c` holds the entries
of the blocks in chunk `c` of TreeAMR's `threadchunks(nblocks(U))` — the
chunk `launch_by_owner!` runs on default-pool thread `c` — one element per
thread, padded with empty ranges where there are fewer blocks than threads.
A block's entries are contiguous in the state vector (`statearray`'s last
index is the block), so each thread owns one range.

`nothing` for a state that is not a CPU `Array`, which is IMEXRungeKutta's
broadcast path: a device array is combined by one fused broadcast on the
device, and a partition would be refused there.

TreeAMR's test suite carries the same helper (`test/imex_tests.jl`) as a
candidate for its own API, and TreeGeneralizedHarmonic a copy; this is the
one this package runs until it is named upstream.
"""
function state_partition(U::FieldSet{T,D}, u) where {T,D}
    u isa Array || return nothing
    L = U.forest.N^D * U.nvars
    length(u) == L * nblocks(U) || throw(DimensionMismatch(
        "the state vector has $(length(u)) entries but the field set's blocks " *
        "hold $(L * nblocks(U)): a partition of one mesh's state cannot be " *
        "applied to another's."))
    parts = UnitRange{Int}[(first(r) - 1) * L + 1:last(r) * L
                           for r in TreeAMR.threadchunks(nblocks(U))]
    while length(parts) < Threads.nthreads()
        push!(parts, 1:0)
    end
    return parts
end

"""
    hydro_integrator(p::HydroProblem, u, t0, t1, nsteps; reset = :stage,
                     alias_u0 = false, reuse = nothing,
                     partition = state_partition(p.U, u))

An IMEXRungeKutta integrator for `nsteps` fixed steps of `SSPRK33` from `t0`
to `t1` on the right-hand side [`hydro_rhs!`](@ref), with the stage
arithmetic split by [`state_partition`](@ref) and the atmosphere reset
installed according to `reset`:

  * `:stage` — [`reset_stage!`](@ref) as the stage limiter, on the two stage
    values per step that the right-hand side reads beyond `uⁿ`, and
    [`reset_atmosphere!`](@ref) as the step limiter, on the step's result.
    Every state the right-hand side reads and every state stored has been
    reset, which is the GRMHD practice and the default.
  * `:step` — [`reset_atmosphere!`](@ref) as the step limiter alone.
  * `:none` — neither.

`alias_u0 = true` makes the integrator step `u` itself; `reuse` hands it an
earlier integrator's scratch, which must be for a state of the same length
and partition (IMEXRungeKutta refuses anything else). [`evolve!`](@ref)
uses both, one integrator per chunk; [`hydro_solve!`](@ref) uses neither.
`partition = nothing` forces the broadcast path on the host, which is what
the benchmark and the test that the two paths agree bit for bit need.

The step count is asserted rather than assumed: IMEXRungeKutta derives it as
`⌈(t1 − t0)/dt⌉` to within a few ulp, and a `dt = (t1 − t0)/nsteps` that
rounded up across an integer would take one step more than the driver
counted and sized its CFL check for.
"""
function hydro_integrator(p::HydroProblem{T}, u, t0::T, t1::T, nsteps::Int;
                          reset::Symbol=:stage, alias_u0::Bool=false,
                          reuse=nothing,
                          partition=state_partition(p.U, u)) where {T}
    check_reset(reset)
    nsteps ≥ 1 || throw(ArgumentError("nsteps must be at least 1, got $nsteps."))
    stage = reset === :stage ? reset_stage! : nothing
    step = reset === :none ? nothing : reset_atmosphere!
    prob = IRK.IMEXProblem(hydro_rhs!, nothing, u, (t0, t1), p)
    integ = IRK.init(prob, IRK.SSPRK33(); dt=(t1 - t0) / nsteps,
                     stage_limiter=stage, step_limiter=step,
                     partition=partition, alias_u0=alias_u0,
                     reuse=reuse)
    integ.nsteps == nsteps || throw(ErrorException(
        "IMEXRungeKutta derived $(integ.nsteps) steps from dt = (t1 − t0)/" *
        "$nsteps; the driver sized its step and its CFL check for $nsteps."))
    return integ
end

"""
    hydro_solve!(p::HydroProblem, u, t0, t1, nsteps; reset = :stage)

One fixed-step `SSPRK33` solve of `nsteps` steps from `t0` to `t1`,
returning the final state vector. `u` is not modified.

Strong-stability-preserving rather than plain Runge–Kutta because a
limited scheme's shocks stay monotone only under one: conservation holds
for *any* Runge–Kutta method, since every stage's `du` already sums to
zero. Fixed step because `λ_max` is measured once per chunk and a
step-adaptive `dt` would be a callback fighting a fixed-step solve; the
driver's CFL recheck at the end of a chunk is what makes that safe. See
"Time integration and the time step" in `CODE.md`.

`reset` chooses the limiter hooks [`hydro_integrator`](@ref) installs:
`:stage` (the default, and GRMHD practice) resets every stage value the
right-hand side reads and every step's result, `:step` only the step's
result, and `:none` nothing. A positivity-preserving correction is what
those hooks exist for, which is the second reason this package wants an
SSP method. `test/reset_tests.jl` asserts that a step actually comes out
floored under `:stage` and under `:step` and does not under `:none`, so the
wiring is checked rather than assumed.

Where nothing is floored the three settings give **bit-identical** results:
the reset writes back only the cells a floor fired in, and the limiter hooks
do not otherwise enter the arithmetic of a stage.
"""
function hydro_solve!(p::HydroProblem{T}, u, t0::T, t1::T, nsteps::Int;
                      reset::Symbol=:stage) where {T}
    integ = hydro_integrator(p, u, t0, t1, nsteps; reset=reset)
    IRK.solve!(integ)
    return integ.u
end
