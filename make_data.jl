using ParticleInCell
using Makie
using DataFrames
using CSV
using FFTW
using ProgressMeter

include("pics.jl")

function run_simulation(sim_func, norm_therm_vel, norm_beam_vel; num_cells=16, norm_dt=1e-3, norm_num_macros=10, num_periods=10, norm_perturb_vel=0.00001, norm_wavenumber=1)
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

    @show sim_func norm_perturb_vel norm_beam_vel norm_therm_vel norm_wavenumber

    # positions = (collect(0:num_macros-1) .+ 0.5) ./ num_macros
    positions = collect(0:num_macros-1) ./ num_macros
    momentums =
        (particles_per_macro * elec_mass) .*
        (perturb_velocity .* sin.(positions .* wavenumber) .+
                thermal_velocity .* randn.() .+
                beam_velocity)

    electrons = ParticleInCell.electrons(positions, momentums, particles_per_macro)

    # Call sim_func to create Simulation struct
    sim, fields = sim_func(grid, electrons, dt)
    phi = fields[:phi]
    Enode = fields[:Enode]
    Eedge = fields[:Eedge]
    rho = fields[:rho]

    # Make vector to store electric field energy
    dump_period = 10
    num_dumps = round(Int64, 2 * pi * num_periods / norm_dt / dump_period)
    num_steps = num_dumps * dump_period
    @show num_dumps num_steps

    dump_times = Vector{Float64}(undef, num_dumps)
    beam_momentum = Vector{Float64}(undef, num_dumps)
    thermal_energy = Vector{Float64}(undef, num_dumps)
    field_energy = Vector{Float64}(undef, num_dumps)
    mode_amp = Vector{Float64}(undef, num_dumps)

    # Run simulation
    sim_time = 0.0
    @showprogress for n = 1:num_steps
        step!(sim)
        sim_time += dt

        # @show rho.values phi.values
        # n > 3 && return

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

function make_algo_data(sim_func, algo_name; norm_num_macros=1000)
    mkpath("data")

    norm_beam_vels = collect(range(0.0, 0.4, step=0.01))
    norm_therm_vels = collect(range(0.0, 0.2, step=0.01))
    for norm_therm_vel = norm_therm_vels, norm_beam_vel = norm_beam_vels
        df = @time run_simulation(sim_func, norm_therm_vel, norm_beam_vel; norm_num_macros, norm_perturb_vel=0.0)

        CSV.write("data/algo=$(algo_name)_bm=$(norm_beam_vel)_tm=$(norm_therm_vel).csv", df)
    end
end
# make_algo_data(mcpic1_sim_func, "mcpic1")
# make_algo_data(ecpic1_sim_func, "ecpic1")
# make_algo_data(ecpic2_sim_func, "ecpic2")
# make_algo_data(ecpic2_new_sim_func, "ecpic2_new")
# make_algo_data(ecpic1_five_sim_func, "ecpic1_five")
# make_algo_data(ecpic2_five_sim_func, "ecpic2_five")
# make_algo_data(ecpic1_five_sim_func, "ecpic1_five", norm_num_macros=10)
# make_algo_data(ecpic2_five_sim_func, "ecpic2_five", norm_num_macros=100)
# make_algo_data(pics_sim_func, "pics")

function make_accuracy_data(sim_func, algo_name)
    mkpath("data")

    norm_wavenumbers = [1, 2, 4, 8, 16, 32, 64]
    # norm_wavenumbers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # norm_wavenumbers = [64, 128, 256]
    for nw in norm_wavenumbers
        # df = @time run_simulation(sim_func, 0., 0.; num_cells=512, norm_dt=1e-4, num_periods=3, norm_wavenumber=nw, norm_num_macros=1, norm_perturb_vel=1e-6)
        # df = @time run_simulation(sim_func, 0., 0.; num_cells=512, norm_dt=1e-5, num_periods=3, norm_wavenumber=nw, norm_num_macros=1, norm_perturb_vel=1e-6)
        df = @time run_simulation(sim_func, 0., 0.; num_cells=128, norm_dt=1e-3, num_periods=3, norm_wavenumber=nw, norm_num_macros=1000, norm_perturb_vel=0.1)

        CSV.write("data/algo=$(algo_name)_nw=$(nw).csv", df)
        # CSV.write("data/algo=$(algo_name)_nw=$(nw)_ppc.csv", df)
        # CSV.write("data/algo=$(algo_name)_nw=$(nw)_fast.csv", df)
        # CSV.write("data/algo=$(algo_name)_nw=$(nw).csv", df)
        # CSV.write("data/algo=$(algo_name)_nw=$(nw)_long.csv", df)

        # CSV.write("data/algo=$(algo_name)_nw=$(nw)_alt.csv", df)
        # CSV.write("data/algo=$(algo_name)_nw=$(nw)_alt2.csv", df)
        # CSV.write("data/algo=$(algo_name)_nw=$(nw)_alt3.csv", df)
    end
end
make_accuracy_data(mcpic1_sim_func, "mcpic1")
make_accuracy_data(ecpic1_sim_func, "ecpic1")
make_accuracy_data(ecpic2_sim_func, "ecpic2")
make_accuracy_data(ecpic2_new_sim_func, "ecpic2_new")
# make_accuracy_data(ecpic1_five_sim_func, "ecpic1_five")
make_accuracy_data(ecpic2_five_sim_func, "ecpic2_five")
make_accuracy_data(pics_sim_func, "pics")

function make_timestep_data(sim_func, algo_name)
    mkpath("data")

    norm_timesteps = [0.1, 0.01, 0.001, 0.0001]
    for nt in norm_timesteps
        df = @time run_simulation(sim_func, 0., 0.; num_cells=128, norm_dt=nt, num_periods=3, norm_wavenumber=1, norm_num_macros=100, norm_perturb_vel=0.1)

        CSV.write("data/algo=$(algo_name)_nt=$(nt).csv", df)
    end
end

# make_timestep_data(mcpic1_sim_func, "mcpic1")
# make_timestep_data(ecpic2_five_sim_func, "ecpic2_five")
