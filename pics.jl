struct PICSFieldSolve <: ParticleInCell.AbstractSimulationStep
    rho0
    rho2

    phi0
    phi2

    Ksq_inv
    c_k

    ft_vector0
    ft_vector2

    fft_plan

    function PICSFieldSolve(
        rho0::F,
        rho2::F,
        phi0::F,
        phi2::F,
    ) where {T,D,G,F<:ParticleInCell.AbstractField{T,D,G}}
        @assert rho0.grid === rho2.grid === phi0.grid === phi2.grid
        @assert all(rho0.grid.periodic)
        @assert num_elements(rho0) == 1 && num_elements(rho2) == 1 && num_elements(phi0) == 1 && num_elements(phi2) == 1

        epsilon_0 = 8.8541878128e-12
        Ksq_inv = zeros(T, rho0.grid.num_cells...)
        c_k = zeros(T, rho0.grid.num_cells...)
        ft_vector0 = zeros(Complex{T}, rho0.grid.num_cells...)
        ft_vector2 = zeros(Complex{T}, rho0.grid.num_cells...)

        grid = rho0.grid
        sim_lengths = grid.upper_bounds .- grid.lower_bounds
        cell_lengths = sim_lengths ./ grid.num_cells
        for I in eachindex(Ksq_inv)
            It = Tuple(I)


            ks =
                2π .*
                ifelse.(It .<= size(Ksq_inv) ./ 2, It .- 1, It .- size(Ksq_inv) .- 1) ./
                sim_lengths

            kdxs = ks .* cell_lengths
            cs = cos.(kdxs)

            ips = (1 ./ cell_lengths.^2) .* (48 .- 9 .* cs .- 36 .* cs.^2 .- 3 .* cs.^3) ./ (5 .* (2 .+ cs).^2)

            inv_Ksqs = 1 ./ (ips .* epsilon_0)
            if any(x -> x == 1, It)
                Ksq_inv[I] = 0
            # elseif any(k -> isapprox(abs(k), pi), ks .* cell_lengths)
            #     Ksq_inv[I] = 0
            else
                Ksq_inv[I] = prod(inv_Ksqs)
            end

            cks = (-12 ./ cell_lengths.^2) .* sin.(ks .* cell_lengths ./ 2) ./ (2 .+ cos.(ks .* cell_lengths))
            c_k[I] = prod(cks)
        end


        fft_plan = plan_fft!(ft_vector0)

        new(rho0, rho2, phi0, phi2, Ksq_inv, c_k, ft_vector0, ft_vector2, fft_plan)
    end
end

function ParticleInCell.step!(step::PICSFieldSolve)
    # step.ft_vector0 .= view(step.rho0.values, eachindex(step.rho0))
    # step.fft_plan * step.ft_vector0
    # step.ft_vector0 .= step.ft_vector0 .* step.Ksq_inv
    # inv(step.fft_plan) * step.ft_vector0
    # view(step.phi0.values, eachindex(step.phi0)) .= real.(step.ft_vector0)

    step.ft_vector0 .= view(step.rho0.values, eachindex(step.rho0))
    step.ft_vector2 .= view(step.rho2.values, eachindex(step.rho2))

    step.fft_plan * step.ft_vector0
    step.fft_plan * step.ft_vector2

    step.ft_vector0 .= (step.ft_vector0 .+ step.ft_vector2 .* step.c_k) .* step.Ksq_inv
    step.ft_vector2 .= step.c_k .* step.ft_vector0

    inv(step.fft_plan) * step.ft_vector0
    inv(step.fft_plan) * step.ft_vector2

    view(step.phi0.values, eachindex(step.phi0)) .= real.(step.ft_vector0)
    view(step.phi2.values, eachindex(step.phi2)) .= real.(step.ft_vector2)
end


# Once the deposition has finished, the result needs to be multiplied by dx^2.
function pics_rho2_deposition(x)
    ax = abs(x)
    if ax > 1
        return zero(x)
    else
        return @evalpoly(ax, 0, -1/3, 1/2, -1/6)
    end
end

function pics_phi0_interpolation(x)
    if x > 1 || x < -1
        return 0
    elseif x < 0
        return 1
    else
        return -1
    end
end

function pics_phi2_interpolation(x)
    if x > 1 || x < -1
        return zero(x)
    elseif x < 0
        return 1/3 + x + x^2 / 2
    else
        return -1/3 + x - x^2 / 2
    end
end

struct PICSParticlePush{S,F,T} <: ParticleInCell.AbstractSimulationStep
    species::S
    phi0::F
    phi2::F
    timestep::T

    function PICSParticlePush(
        species::S,
        phi0::F,
        phi2::F,
        timestep::T,
    ) where {S,F,T}
        new{S,F,T}(
            species,
            phi0,
            phi2,
            timestep,
        )
    end
end

function ParticleInCell.step!(step::PICSParticlePush)
    species = step.species

    for n in eachindex(species)
        # Push the particle based on its current velocity
        particle_position!(
            species,
            n,
            particle_position(species, n) .+
            (step.timestep / particle_mass(species, n)) .* particle_momentum(species, n),
        )

        # Accelerate the particle according to E
        # Find which cell the particle is in, and create a CartesianIndices
        # object that extends +/- interpolation_width in all directions
        particle_cell_coord, Is = ParticleInCell.phys_coords_to_cell_index_ittr(
            step.phi0,
            particle_position(species, n),
            1,
        )

        for I in Is
            grid_cell_coord = ParticleInCell.cell_index_to_cell_coords(step.phi0, I, 1)
            dist = Tuple(particle_cell_coord .- grid_cell_coord)

            phi0_interp_weights = pics_phi0_interpolation.(dist)
            phi0_interp_weight = prod(phi0_interp_weights)
            phi2_interp_weights = pics_phi2_interpolation.(dist)
            phi2_interp_weight = prod(phi2_interp_weights)

            dx = first(ParticleInCell.cell_lengths(step.phi0.grid))

            particle_momentum!(
                species,
                n,
                particle_momentum(species, n) .-
                ((phi0_interp_weight * step.timestep * particle_charge(species, n) / dx) .*
                    step.phi0.values[I]) .-
                ((phi2_interp_weight * step.timestep * particle_charge(species, n) * dx) .*
                    step.phi2.values[I]),
            )
        end
    end
end


function pics_sim_func(grid, electrons, dt)
    sim = Simulation(ParticleInCell.AbstractSimulationStep[])

    dx = first(ParticleInCell.cell_lengths(grid))

    lower_guard_cells = 2
    rho0 = Field(grid, NodeOffset(), 1, lower_guard_cells)
    rho2 = Field(grid, NodeOffset(), 1, lower_guard_cells)
    phi0 = Field(grid, NodeOffset(), 1, lower_guard_cells)
    phi2 = Field(grid, NodeOffset(), 1, lower_guard_cells)
    Eedge = Field(grid, EdgeOffset(), 1, lower_guard_cells)
    Enode = Field(grid, NodeOffset(), 1, lower_guard_cells)

    push!(sim.steps, ParticleInCell.ZeroField(rho0))
    push!(sim.steps, BSplineChargeInterpolation(electrons, rho0,  1))
    push!(sim.steps, CommunicateGuardCells(rho0, true))

    push!(sim.steps, ParticleInCell.ZeroField(rho2))
    push!(sim.steps, BSplineChargeInterpolation(electrons, rho2,  1, pics_rho2_deposition))
    push!(sim.steps, ParticleInCell.MultiplyField(rho2, dx^2))
    push!(sim.steps, CommunicateGuardCells(rho2, true))

    push!(sim.steps, PICSFieldSolve(rho0, rho2, phi0, phi2))
    push!(sim.steps, CommunicateGuardCells(phi0))
    push!(sim.steps, CommunicateGuardCells(phi2))

    push!(sim.steps, PICSParticlePush(electrons, phi0, phi2, dt))
    push!(sim.steps, CommunicateSpecies(electrons, grid))

    return sim, (; rho=rho0, phi=phi0, phi2, rho2, Eedge, Enode)
end
