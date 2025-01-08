@info "Driver script started"

@info "Adding worker processes"
using Distributed
using SlurmClusterManager

if "SLURM_JOB_ID" in keys(ENV)
    addprocs(SlurmManager())
else
    addprocs(8)
end

@info "Activating project on worker processes"
@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@info "Loading simulation code on worker processes"
@everywhere include("make_data.jl")

function make_algo_data(sim_func, algo_name; norm_num_macros=2^14, num_cells=16, init_strat="beam")
    mkpath("data")

    norm_beam_vels = collect(range(0.0, 0.45, step=0.01))
    norm_therm_vels = collect(range(0.0, 0.25, step=0.01))

    @sync @distributed for (norm_beam_vel, norm_therm_vel) = collect(Iterators.product(norm_beam_vels, norm_therm_vels))
        df = run_simulation(sim_func, norm_therm_vel, norm_beam_vel; norm_num_macros, norm_perturb_vel=1e-8, num_periods=100, init_strat, perturb_all_wavenumbers=true, num_cells)

        CSV.write("data/algo=$(algo_name)_bm=$(norm_beam_vel)_tm=$(norm_therm_vel).csv", df)
    end
end
make_algo_data(mcpic1_sim_func, "mcpic1")
make_algo_data(ecpic1_sim_func, "ecpic1")
make_algo_data(ecpic2_sim_func, "ecpic2")
make_algo_data(ecpic2_new_sim_func, "ecpic2_new")
make_algo_data(ecpic2_five_sim_func, "ecpic2_five")
make_algo_data(pics_sim_func, "pics")

function make_stationary_algo_data(sim_func, algo_name; norm_num_macros=1000, num_cells=16, norm_dt=0.1, init_strat="quiet")
    mkpath("data")

    norm_therm_vels = collect(range(0.0, 0.3, step=0.01))
    # norm_therm_vels = collect(range(0.16, 0.2, step=0.01))
    # norm_therm_vels = [0.16, 0.17, 0.15]
    for norm_therm_vel = norm_therm_vels
        # df = @time run_simulation(sim_func, norm_therm_vel, 0.0; norm_num_macros, norm_perturb_vel=0.0, norm_dt=1.0, num_periods=50, num_cells=8)
        df = @time run_simulation(sim_func, norm_therm_vel, 0.0; norm_num_macros, norm_perturb_vel=0.0, norm_dt, num_periods=100, num_cells, init_strat)

        CSV.write("data/algo=$(algo_name)_bm=0.0_tm=$(norm_therm_vel)_ppc=$(norm_num_macros)_init_strat=$(init_strat).csv", df)
    end
end
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^6)
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^8)
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^10)
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^12)
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^14)
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^16)
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^18, norm_dt=0.5)
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^20, norm_dt=0.5, num_cells=8)

# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^16, init_strat="quiet")
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^16, init_strat="noisy")
# make_stationary_algo_data(mcpic1_sim_func, "mcpic1", norm_num_macros=2^16, init_strat="beam")
