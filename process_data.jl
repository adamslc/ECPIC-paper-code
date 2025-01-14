using CSV
using DataFrames
using FFTW
using CairoMakie
using Colors
using ColorSchemes
using GLM

function read_data(algo, bm, tm)
    CSV.read("data/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
    # CSV.read("data20240419/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
end

function compute_fit(df; growth_cutoff=1e-2)
    ynorm = df[1, :thermal_energy]
    df[:, :norm_thermal_energy] = (df[!, :thermal_energy] .- ynorm) ./ ynorm


    imax = findfirst(x -> x > growth_cutoff, df[!, :norm_thermal_energy])
    if imax === nothing
        imax = size(df, 1)
    end

    imin = 1 + findlast(x -> x <= 0, df[!, :norm_thermal_energy])
    if imin === nothing
        imin = 1
    end

    if imin > imax
        return imin:imax, nothing
    end

    df[:, :log_norm_thermal_energy] = log.(abs.(df[!, :norm_thermal_energy]))

    fit = lm(@formula(log_norm_thermal_energy ~ 1 + norm_time), df[imin:imax, :])

    return imin:imax, fit
end

@. exp_model(x, p) = exp((p[2] * x) + p[1])

function make_combo_fit_plot_axis(ax, df; show_fit=true, growth_cutoff=1e-2)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.ylabel = L"|\Delta E_\text{th}| / E_\text{th}(0)"

    ind_range, fit = compute_fit(df; growth_cutoff)

    color_val = sign.(df[!, :norm_thermal_energy])
    cs = ColorScheme([colorant"blue", colorant"black"])
    lines!(ax, df[!, :norm_time] ./ (2pi), abs.(df[!, :norm_thermal_energy]), color=color_val, colormap=cs, linewidth=2)

    if show_fit && fit !== nothing
        xs = df[ind_range, :norm_time]
        ys = exp_model(xs, coef(fit))

        scatterlines!(ax, xs ./ (2pi), ys, color=:red, linewidth=2, markersize=2)
        text!(ax, 0, 1, text = L"r^2 = %$(round(r2(fit), digits=3))", space=:relative, align=(:left, :top))
    end

    hlines!(ax, [growth_cutoff], color=:black, linestyle=:dash)
end

function make_fit_plot(df; show_fit=true, growth_cutoff=1e-2, range=(-15,2), filename="fit.pdf")
    fig = Figure(size=(325, 220), fonts=(; regular="Times New Roman"), fontsize=14)

    ax1 = Axis(fig[1, 1], yscale=log10)
    make_combo_fit_plot_axis(ax1, df; show_fit, growth_cutoff)

    ax1.xlabel = L"t \omega_p / 2 \pi"
    ax1.limits = (nothing, (10.).^(range))
    save(filename, fig)
end

function make_combo_fit_plot(; growth_cutoff=1e-2)
    fig = Figure(size=(325, 250), fonts=(; regular="Times New Roman"), fontsize=14)

    @info "First"
    ax1 = Axis(fig[1, 1], yscale=log10)
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.01.csv", DataFrame)
    make_combo_fit_plot_axis(ax1, df; growth_cutoff)
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false
    ax1.limits = ((0, 100), (1e-6, 1e2))

    @info "Second"
    ax2 = Axis(fig[2, 1], yscale=log10)
    # ax2 = Axis(fig[2, 1]) 
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.1.csv", DataFrame)
    make_combo_fit_plot_axis(ax2, df; growth_cutoff)
    ax2.xlabel = L"t \omega_p / 2 \pi"
    ax2.limits = ((0, 100), (1e-6, 1e2))

    save("combo_fit.pdf", fig)
end

function compute_growth_rate(df; growth_cutoff=1e-2, r2_cutoff=0.5)
    ind_range, fit = compute_fit(df; growth_cutoff)

    if fit === nothing
        return 0.0
    end

    if (df[last(ind_range), :norm_time] - df[first(ind_range), :norm_time]) / (2pi) < 10
        if df[end, :norm_thermal_energy] < growth_cutoff
            return 0.0
        end
    end

    if r2(fit) < r2_cutoff
        return 0.0
    end

    growth_rate = max(coef(fit)[2] / 2, 0)

    if isnan(growth_rate)
        return 0.0
    end

    return growth_rate
end

function compute_growth_rates(algo; growth_cutoff=1e-2, r2_cutoff=0.9)
    norm_beam_vels = collect(range(0.0, 0.45, step=0.01))
    norm_therm_vels = collect(range(0.01, 0.25, step=0.01))

    growth_rates = Matrix{Float64}(undef, length(norm_therm_vels), length(norm_beam_vels))

    for (i, norm_therm_vel) = enumerate(norm_therm_vels), (j, norm_beam_vel) = enumerate(norm_beam_vels)
        try
            df = read_data(algo, norm_beam_vel, norm_therm_vel)
            growth_rate = compute_growth_rate(df; growth_cutoff, r2_cutoff)

            if growth_rate < 0
                @warn "Negative growth rate" norm_therm_vel norm_beam_vel growth_rate
            end

            growth_rates[i, j] = growth_rate
        catch err
            @error "Error measuring growth rate" algo norm_therm_vel norm_beam_vel err
            growth_rates[i, j] = +Inf
        end
    end

    return norm_beam_vels, norm_therm_vels, growth_rates
end

function make_growth_axis(ax, algo; hidex=false, hidey=false, colorrange=(-2, 0), v_critical=nothing, growth_cutoff=1e-2, r2_cutoff=0.9)
    @info "Making growth axis" algorithm=algo hidex hidey

    norm_beam_vels, norm_therm_vels, growth_rates = compute_growth_rates(algo; growth_cutoff, r2_cutoff)

    hm = heatmap!(ax, norm_beam_vels, norm_therm_vels, log10.(transpose(growth_rates)); colorrange, colormap=:inferno, lowclip=:black, highclip=:white)

    ax.xlabel = L"\bar{v}_d"
    ax.ylabel = L"\bar{v}_t"
    ax.ylabelrotation = 0

    ax.xticks = ([0.1, 0.2, 0.3, 0.4, 0.5], [L"0.1", L"0.2", L"0.3", L"0.4", L"0.5"])
    ax.yticks = ([0.0, 0.1, 0.2, 0.3], [L"0.0", L"0.1", L"0.2", L"0.3"])
    ax.leftspinevisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    ax.bottomspinevisible = false

    if hidex
        ax.xlabelvisible = false
        # ax.xticklabelsvisible = false
        # ax.xticksvisible = false
    end

    if hidey
        ax.ylabelvisible = false
        ax.yticklabelsvisible = false
        # ax.yticksvisible = false
    end

    if !isnothing(v_critical)
        ax.xminorticks = [v_critical]
        ax.xminortickcolor = :red
        ax.xminorticksvisible = true
        ax.xminorticksize = 7
        ax.xminortickwidth = 1
    end

    return hm
end

function make_growth_plot(algo; title=algo, growth_cutoff=1e-2)

    fig = Figure()

    ax1 = Axis(fig[1,1], aspect=DataAspect(), title=title)
    hm = make_growth_axis(ax1, algo; growth_cutoff)

    cbar = Colorbar(fig[:, 2], hm, label=L"\text{(Growth rate)} / \omega_p")
    cbar.ticks = ([-3, -2, -1, 0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])

    save("$(algo).pdf", fig)
end

# make_growth_plot("mcpic1", growth_cutoff=-12)
# make_growth_plot("ecpic1", growth_cutoff=-5)
# make_growth_plot("ecpic2", growth_cutoff=-10)
# make_growth_plot("ecpic2_new"; title="ecpic2 w/ 5 pt field solve", growth_cutoff=-8)
# make_growth_plot("ecpic2_five"; title="ecpic2 w/ 4th order solve", growth_cutoff=-10)
# make_growth_plot("pics", growth_cutoff=-5)

function make_combo_growth_heatmap(; growth_cutoff=1e-2, r2_cutoff=0.9)
    # 624 units corresponds to a width of 6.5 inches
    # fig = Figure(size=(624, 400))
    fig = Figure(size=(640, 400))

    ax1 = Axis(fig[1,1], aspect=DataAspect(), title="MC-PIC1")
    hm = make_growth_axis(ax1, "mcpic1"; hidex=true, growth_cutoff, r2_cutoff)

    ax2 = Axis(fig[1,2], aspect=DataAspect(), title="EC-PIC1")
    make_growth_axis(ax2, "ecpic1"; hidex=true, hidey=true, v_critical=0.288, growth_cutoff, r2_cutoff)
    # vlines!(ax2, [0.35], color=:white, linewidth=1, linestyle=:dash)

    ax3 = Axis(fig[1,3], aspect=DataAspect(), title="EC-PIC2-Standard")
    make_growth_axis(ax3, "ecpic2"; hidex=true, hidey=true, v_critical=0.183, growth_cutoff, r2_cutoff)

    ax4 = Axis(fig[2,1], aspect=DataAspect(), title="EC-PIC2-Fourth")
    hm = make_growth_axis(ax4, "ecpic2_five"; v_critical=0.158, growth_cutoff, r2_cutoff)

    ax5 = Axis(fig[2,2], aspect=DataAspect(), title="EC-PIC2-Lagrange")
    make_growth_axis(ax5, "ecpic2_new"; hidey=true, v_critical=0.316, growth_cutoff, r2_cutoff)

    ax6 = Axis(fig[2,3], aspect=DataAspect(), title="CS-PIC")
    make_growth_axis(ax6, "pics"; hidey=true, growth_cutoff, r2_cutoff)

    cbar = Colorbar(fig[:, 4], hm, label=L"\text{(Growth rate)} / \omega_p")
    # cbar.ticks = ([-3, -2, -1, 0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
    # cbar.ticks = ([-5, -4, -3, -2, -1, 0], [L"10^{-5}",L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
    cbar.ticks = ([-3, -2, -1, 0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])

    save("combo_growth_heatmap.pdf", fig)
end

function compute_stationary_growth_rates(algo, ppc; norm_therm_vels = collect(range(0.01, 0.3, step=0.01)), num_bins=10, growth_cutoff=-2, init_strat="beam")
    growth_rates = Vector{Float64}(undef, length(norm_therm_vels))

    for (i, norm_therm_vel) = enumerate(norm_therm_vels)
        try
            df = CSV.read("data/algo=$(algo)_bm=0.0_tm=$(norm_therm_vel)_ppc=$(ppc)_init_strat=$(init_strat).csv", DataFrame)
            # df = CSV.read("data/algo=$(algo)_bm=0.0_tm=$(norm_therm_vel)_ppc=$(ppc).csv", DataFrame)
            # df = CSV.read("data20241213/algo=$(algo)_bm=0.0_tm=$(norm_therm_vel)_ppc=$(ppc).csv", DataFrame)
            growth_rate = compute_max_growth_rate(df; num_bins, growth_cutoff)

            if growth_rate < 0
                @warn "Negative growth rate" norm_therm_vel growth_rate
            end

            growth_rates[i] = growth_rate
        catch err
            @error "Error measuring growth rate" algo norm_therm_vel err
            growth_rates[i] = 0
        end
    end

    return norm_therm_vels, growth_rates
end

function stationary_stab_plot(algo; growth_cutoff=-2)
    fig = Figure(size=(325, 300), fonts=(; regular="Times New Roman"), fontsize=14)

    ax1 = Axis(fig[1, 1], yscale=log10)

    # ppcs = [1000, 10000, 100000, 1000000]
    # limits = [-1, -2, -3, -4]
    # ppcs = [2^10, 2^12, 2^14, 2^16, 2^18, 2^20]
    # limits = [-3, -3, -3, -3, -3, -6]
    ppcs = [2^16]
    limits = [-6]

    for (ppc, limit) in zip(ppcs, limits)
        norm_therm_vels, growth_rates = compute_stationary_growth_rates(algo, ppc; growth_cutoff=limit)
        # growth_rates = [g == 0 ? 1e-9 : g for g in growth_rates]
        # @info "Growth rates" ppc limit norm_therm_vels growth_rates
        ppc_exp10 = round(log10(ppc), digits=1)
        ppc_exp2 = round(Int, log2(ppc))
        lines!(ax1, norm_therm_vels, growth_rates, label=L"\textrm{ppc}=2^{%$(ppc_exp2)} \approx 10^{%$(ppc_exp10)}")
    end

    ax1.xlabel = L"\bar{v}_t"
    ax1.ylabel = L"\gamma / \omega_p"

    ax1.limits = ((0.0, 0.3), (1e-3, 1e-1))

    axislegend(ax1)

    save("stationary_$(algo).pdf", fig)
end

growth_cutoff = 0.01
r2_cutoff = 0.75
# make_combo_fit_plot(; growth_cutoff)
make_combo_growth_heatmap(; growth_cutoff, r2_cutoff)
# stationary_stab_plot("mcpic1"; growth_cutoff)

# Noise, but with an r value above 0.95
# df = read_data("ecpic1", 0.45, 0.21)
# df = read_data("ecpic1", 0.45, 0.22)
# df = read_data("ecpic1", 0.33, 0.22)
# df = read_data("ecpic1", 0.33, 0.21)
# df = read_data("ecpic1", 0.33, 0.2)
# df = read_data("ecpic1", 0.33, 0.19)
# df = read_data("ecpic1", 0.33, 0.18)
# df = read_data("ecpic1", 0.33, 0.17)
# ind_range, fit = compute_fit(df)
# @show ind_range fit
# make_fit_plot(df)

function lineout_fit_plot(sims; growth_cutoff=1e-2, range=(-15,2))
    height = length(sims) * 100
    fig = Figure(size=(325, height), fonts=(; regular="Times New Roman"), fontsize=14)

    for (i, sim) in enumerate(sims)
        name, df = sim
        ax = Axis(fig[i, 1], yscale=log10, xticklabelsvisible=false, xticksvisible=false)
        make_combo_fit_plot_axis(ax, df; growth_cutoff)
        text!(ax, 1, 1, text = name, space=:relative, align=(:right, :top))
        ax.limits = (nothing, (10.).^(range))
    end

    save("fit_lineout.pdf", fig)
end

function read_data2(algo, bm, tm)
    return (L"\bar{v}_t = %$(tm)", read_data(algo, bm, tm))
end

dfs = [read_data2("mcpic1", 0.0, tm) for tm in range(0.25, 0.01, step=-0.01)]
lineout_fit_plot(dfs)
