using ParticleInCell
using Makie
using DataFrames
using CSV
using FFTW
using ProgressMeter
using Distributions
using Distributed

include("pics.jl")

# Taken from ChatGPT, with modifications
function reverse_bits(x::Int, nbits::Int)
    result = 0
    for i in 1:nbits
        result |= ((x >> (i - 1)) & 1) << (nbits - i)
    end
    return result
end

function bit_reverse_permutation!(arr::Vector)
    @assert ispow2(length(arr)) "Length of array must be a power of 2"

    n = length(arr)
    m = trailing_zeros(nextpow(2, n)) # Number of bits required
    reversed_indices = [reverse_bits(i, m) + 1 for i in 0:n-1] # Compute bit-reversed indices

    permute!(arr, reversed_indices)
end

function run_simulation(sim_func, norm_therm_vel, norm_beam_vel; num_cells=16, norm_dt=0.1, norm_num_macros=10, num_periods=10, norm_perturb_vel=0.00001, norm_wavenumber=1, init_strat="quiet")
    # Create grid
    sim_length = 1.0
    grid = UniformCartesianGrid((0.0,), (sim_length,), (num_cells,), (true,))

    # Create species
    number_density = 1e14
    epsilon_0 = 8.8541878128e-12
    elec_charge = 1.60217663e-19
    elec_mass = 9.1093837e-31
    plasma_freq = sqrt(number_density * elec_charge^2 / elec_mass / epsilon_0)
    dt = norm_dt / plasma_freq

    dx = sim_length / num_cells
    num_macros = num_cells * norm_num_macros
    particles_per_macro = number_density * sim_length / num_macros

    wavenumber = norm_wavenumber * 2 * pi / sim_length
    perturb_velocity = norm_perturb_vel * plasma_freq * dx
    thermal_velocity = norm_therm_vel * plasma_freq * dx
    beam_velocity = norm_beam_vel * plasma_freq * dx

    # Make vector to store electric field energy
    dump_period = 10
    num_dumps = round(Int64, 2 * pi * num_periods / norm_dt / dump_period)
    num_steps = num_dumps * dump_period

    dist = Normal(beam_velocity, thermal_velocity)

    positions = Vector{Float64}(undef, num_macros)
    momentums = Vector{Float64}(undef, num_macros)

    if init_strat == "noisy"
        positions .= collect(0:num_macros-1) ./ num_macros
        momentums .= (particles_per_macro * elec_mass) .* rand.(dist)
    elseif init_strat == "quiet"
        positions .= bit_reverse_permutation!(collect(0:num_macros-1) ./ num_macros)
        momentums .= (particles_per_macro * elec_mass) .* quantile.(dist, (0.5 .+ collect(0:num_macros - 1)) ./ num_macros)
    elseif init_strat == "beam"
        positions_partial = bit_reverse_permutation!(collect(0:norm_num_macros-1) ./ norm_num_macros .* dx)
        momentums_partial = (particles_per_macro * elec_mass) .* quantile.(dist, (0.5 .+ collect(0:norm_num_macros - 1)) ./ norm_num_macros)

        for i in 1:norm_num_macros
            for j in 1:num_cells
                positions[i + (j - 1) * norm_num_macros] = positions_partial[i] + (j - 1) * dx
                momentums[i + (j - 1) * norm_num_macros] = momentums_partial[i]
            end
        end
    else
        throw(ArgumentError("Invalid init_strat"))
    end

    # Add velocity perturbation at wavenumber with a random phase
    phase = 2pi * rand()
    momentums .+= (particles_per_macro * elec_mass * perturb_velocity) .* cos.(wavenumber .* positions .+ phase)

    electrons = ParticleInCell.electrons(positions, momentums, particles_per_macro)

    # Call sim_func to create Simulation struct
    sim, fields = sim_func(grid, electrons, dt)
    phi = fields[:phi]
    Enode = fields[:Enode]
    Eedge = fields[:Eedge]
    rho = fields[:rho]


    dump_times = Vector{Float64}(undef, num_dumps)
    beam_momentum = Vector{Float64}(undef, num_dumps)
    thermal_energy = Vector{Float64}(undef, num_dumps)
    field_energy = Vector{Float64}(undef, num_dumps)
    mode_amp = Vector{Float64}(undef, num_dumps)

    start_time = time()

    @info "Running simulation with parameters:" sim_func init_strat norm_beam_vel norm_therm_vel norm_perturb_vel norm_wavenumber num_cells norm_num_macros norm_dt num_periods num_dumps num_steps myid()

    # Run simulation
    sim_time = 0.0
    # @showprogress for n = 1:num_steps
    for n = 1:num_steps
        step!(sim)
        sim_time += dt

        if n % dump_period == 0
            dump_number = div(n, dump_period)

            dump_times[dump_number] = sim_time * plasma_freq
            beam_momentum[dump_number] = 0
            thermal_energy[dump_number] = 0
            field_energy[dump_number] = 0
            for I in eachindex(electrons.momentums)
                beam_momentum[dump_number] += electrons.momentums[I][1]
            end
            beam_momentum[dump_number] /= num_macros
            for I in eachindex(electrons.momentums)
                thermal_energy[dump_number] += (electrons.momentums[I][1] - beam_momentum[dump_number])^2 /
                    2 / particles_per_macro / elec_mass
            end
            for I in eachindex(rho)
                field_energy[dump_number] += rho[I] * phi[I] * dx / 2
            end

            amps = real.(fft(phi[eachindex(phi)]))
            mode_amp[dump_number] = amps[1 + norm_wavenumber]
        end
    end

    @info "Finished running with parameters:" sim_func init_strat norm_beam_vel norm_therm_vel norm_perturb_vel norm_wavenumber num_cells norm_num_macros norm_dt num_periods num_dumps num_steps myid() elapsed_time=time()-start_time

    df = DataFrame(
        "norm_time" => dump_times,
        "thermal_energy" => thermal_energy,
        "beam_energy" => beam_momentum.^2 ./ (2 * particles_per_macro * elec_mass / num_macros),
        "field_energy" => field_energy,
        "mode_amp" => mode_amp)
    return df
end

mcpic1_sim_func(grid, electrons, dt) = create_electrostatic_simulation(grid, [electrons], dt)

function ecpic1_sim_func(grid, electrons, dt)
    sim = Simulation(ParticleInCell.AbstractSimulationStep[])

    # Create fields
    # This is currently set to 2 to work around a bug in how interpolation cell
    # ranges are calculated for the 2nd order field interpolation. It can be reset
    # to 1 once that bug is resolved.
    lower_guard_cells = 2
    rho = Field(grid, NodeOffset(), 1, lower_guard_cells)
    phi = Field(grid, NodeOffset(), 1, lower_guard_cells)
    Eedge = Field(grid, EdgeOffset(), 1, lower_guard_cells)
    Enode = Field(grid, NodeOffset(), 1, lower_guard_cells)

    # Zero out charge density then deposit and communicate rho
    push!(sim.steps, ParticleInCell.ZeroField(rho))
    push!(sim.steps, BSplineChargeInterpolation(electrons, rho,  1))
    push!(sim.steps, CommunicateGuardCells(rho, true))

    # Field solve and communicate phi
    push!(sim.steps, PoissonSolveFFT(rho, phi))
    push!(sim.steps, CommunicateGuardCells(phi))

    # Calculate Eedge and communicate
    push!(sim.steps, FiniteDifferenceToEdges(phi, Eedge))
    push!(sim.steps, CommunicateGuardCells(Eedge))

    # Push particles for each species
    push!(sim.steps, ElectrostaticParticlePush(electrons, Eedge, dt, 0))
    push!(sim.steps, CommunicateSpecies(electrons, grid))

    return sim, (; rho, phi, Eedge, Enode)
end

function ecpic2_sim_func(grid, electrons, dt)
    sim = Simulation(ParticleInCell.AbstractSimulationStep[])

    # Create fields
    lower_guard_cells = 3
    rho = Field(grid, NodeOffset(), 1, lower_guard_cells)
    phi = Field(grid, NodeOffset(), 1, lower_guard_cells)
    Eedge = Field(grid, EdgeOffset(), 1, lower_guard_cells)
    Enode = Field(grid, NodeOffset(), 1, lower_guard_cells)

    # Zero out charge density then deposit and communicate rho
    push!(sim.steps, ParticleInCell.ZeroField(rho))
    push!(sim.steps, BSplineChargeInterpolation(electrons, rho,  2))
    push!(sim.steps, CommunicateGuardCells(rho, true))

    # Field solve and communicate phi
    push!(sim.steps, PoissonSolveFFT(rho, phi))
    push!(sim.steps, CommunicateGuardCells(phi))

    # Calculate Eedge and communicate
    push!(sim.steps, FiniteDifferenceToEdges(phi, Eedge))
    push!(sim.steps, CommunicateGuardCells(Eedge))

    # Push particles for each species
    push!(sim.steps, ElectrostaticParticlePush(electrons, Eedge, dt, 1))
    push!(sim.steps, CommunicateSpecies(electrons, grid))

    return sim, (; rho, phi, Eedge, Enode)
end

function ecpic2_new_sim_func(grid, electrons, dt)
    sim = Simulation(ParticleInCell.AbstractSimulationStep[])

    # Create fields
    lower_guard_cells = 3
    rho = Field(grid, NodeOffset(), 1, lower_guard_cells)
    phi = Field(grid, NodeOffset(), 1, lower_guard_cells)
    Eedge = Field(grid, EdgeOffset(), 1, lower_guard_cells)
    Enode = Field(grid, NodeOffset(), 1, lower_guard_cells)

    # Zero out charge density then deposit and communicate rho
    push!(sim.steps, ParticleInCell.ZeroField(rho))
    push!(sim.steps, BSplineChargeInterpolation(electrons, rho,  2))
    push!(sim.steps, CommunicateGuardCells(rho, true))

    # Field solve and communicate phi
    push!(sim.steps, ParticleInCell.PoissonSolveFFT(rho, phi, ParticleInCell.field_solve_lagrange))
    push!(sim.steps, CommunicateGuardCells(phi))

    # Calculate Eedge and communicate
    push!(sim.steps, FiniteDifferenceToEdges(phi, Eedge))
    push!(sim.steps, CommunicateGuardCells(Eedge))

    # Push particles for each species
    push!(sim.steps, ElectrostaticParticlePush(electrons, Eedge, dt, 1))
    push!(sim.steps, CommunicateSpecies(electrons, grid))

    return sim, (; rho, phi, Eedge, Enode)
end

function ecpic1_five_sim_func(grid, electrons, dt)
    sim = Simulation(ParticleInCell.AbstractSimulationStep[])

    # Create fields
    lower_guard_cells = 3
    rho = Field(grid, NodeOffset(), 1, lower_guard_cells)
    phi = Field(grid, NodeOffset(), 1, lower_guard_cells)
    Eedge = Field(grid, EdgeOffset(), 1, lower_guard_cells)
    Enode = Field(grid, NodeOffset(), 1, lower_guard_cells)

    # Zero out charge density then deposit and communicate rho
    push!(sim.steps, ParticleInCell.ZeroField(rho))
    push!(sim.steps, BSplineChargeInterpolation(electrons, rho,  1))
    push!(sim.steps, CommunicateGuardCells(rho, true))

    # Field solve and communicate phi
    push!(sim.steps, ParticleInCell.PoissonSolveFFT(rho, phi, ParticleInCell.field_solve_five_pt))
    push!(sim.steps, CommunicateGuardCells(phi))

    # Calculate Eedge and communicate
    push!(sim.steps, FiniteDifferenceToEdges(phi, Eedge))
    push!(sim.steps, CommunicateGuardCells(Eedge))

    # Push particles for each species
    push!(sim.steps, ElectrostaticParticlePush(electrons, Eedge, dt, 0))
    push!(sim.steps, CommunicateSpecies(electrons, grid))

    return sim, (; rho, phi, Eedge, Enode)
end

function ecpic2_five_sim_func(grid, electrons, dt)
    sim = Simulation(ParticleInCell.AbstractSimulationStep[])

    # Create fields
    lower_guard_cells = 3
    rho = Field(grid, NodeOffset(), 1, lower_guard_cells)
    phi = Field(grid, NodeOffset(), 1, lower_guard_cells)
    Eedge = Field(grid, EdgeOffset(), 1, lower_guard_cells)
    Enode = Field(grid, NodeOffset(), 1, lower_guard_cells)

    # Zero out charge density then deposit and communicate rho
    push!(sim.steps, ParticleInCell.ZeroField(rho))
    push!(sim.steps, BSplineChargeInterpolation(electrons, rho,  2))
    push!(sim.steps, CommunicateGuardCells(rho, true))

    # Field solve and communicate phi
    push!(sim.steps, ParticleInCell.PoissonSolveFFT(rho, phi, ParticleInCell.field_solve_five_pt))
    push!(sim.steps, CommunicateGuardCells(phi))

    # Calculate Eedge and communicate
    push!(sim.steps, FiniteDifferenceToEdges(phi, Eedge))
    push!(sim.steps, CommunicateGuardCells(Eedge))

    # Push particles for each species
    push!(sim.steps, ElectrostaticParticlePush(electrons, Eedge, dt, 1))
    push!(sim.steps, CommunicateSpecies(electrons, grid))

    return sim, (; rho, phi, Eedge, Enode)
end

