using CSV
using DataFrames
using LsqFit
using FFTW
using CairoMakie

function read_data(algo, bm, tm)
    CSV.read("data/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
    # CSV.read("data20240419/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
end

function compute_fit(xs, ys; growth_cutoff=-3)
    @. model(x, p) = (p[1] * x) + p[2]

    log_ys = Vector{Float64}()
    new_xs = Vector{Float64}()
    for i in eachindex(ys)
        log_y = log(abs(ys[i]))
        if !(isnan(log_y) || isinf(log_y))
            push!(new_xs, xs[i])
            push!(log_ys, log_y)
        end
    end

    fit = curve_fit(model, new_xs, log_ys, [1e-12, log_ys[1]])

    if log10(ys[end]) < growth_cutoff
        @info "Growth rate too small" coef(fit) log10(ys[end]) log_ys[end]
        return [0.0, 0.0]
    end

    return coef(fit)
end

function compute_fits(df, num_bins=20; growth_cutoff=-3)
    fits = Vector{Vector{Float64}}(undef, num_bins)

    for i in 1:num_bins
        si = round(Int, (i - 1)*length(df[!, :norm_time]) / num_bins) + 1
        ei = min(round(Int, i*length(df[!, :norm_time]) / num_bins) + 1, length(df[!, :norm_time]))

        # fit = compute_fit(df[si:ei, :norm_time], df[si:ei, :thermal_energy])
        fit = compute_fit(df[si:ei, :norm_time], (df[si:ei, :thermal_energy] .- df[1, :thermal_energy]) ./ df[1, :thermal_energy]; growth_cutoff)

        fits[i] = fit
    end

    return fits
end

@. exp_model(x, p) = exp((p[1] * x) + p[2])

function make_fit_plot(df; num_bins=10, show_fits=false)
    # fig, ax, plt = lines(df[!, :norm_time], log10.(df[!, :thermal_energy]), color=:blue)
    fig, ax, plt = lines(df[!, :norm_time], abs.(df[!, :thermal_energy] .- df[1, :thermal_energy]) ./ df[1, :thermal_energy], color=:blue)

    ax.limits = (nothing, (1e-6, nothing))
    ax.yscale = log10

    if show_fits
        fits = compute_fits(df, num_bins)
        for i in 1:num_bins
            si = round(Int, (i - 1)*length(df[!, :norm_time]) / num_bins) + 1
            ei = min(round(Int, i*length(df[!, :norm_time]) / num_bins) + 1, length(df[!, :norm_time]))

            fit = fits[i]
            println(fit)

            xs = [df[si, :norm_time], df[ei, :norm_time]]
            ys = exp_model(xs, fit)

            lines!(ax, xs, ys, color=:black)
        end
    end

    # lines!(ax, df[!, :norm_time], df[!, :field_energy], color=:red)
    # lines!(ax, df[!, :norm_time], df[!, :beam_energy], color=:green)
    # lines!(ax, df[!, :norm_time], df[!, :beam_energy] + df[!, :thermal_energy] + df[!, :field_energy], color=:black)

    save("fit.pdf", fig)
    # save("fit.png", fig)
end

function make_combo_fit_plot_axis(ax, df; num_bins=10, show_fits=true, growth_cutoff=-3)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.ylabel = L"|\Delta E_\text{th}| / E_\text{th}(0)"

    ynorm = df[1, :thermal_energy]

    lines!(ax, df[!, :norm_time] ./ (2pi), abs.(df[!, :thermal_energy] .- ynorm) ./ ynorm, color=:black, linewidth=2)

    if show_fits
        fits = compute_fits(df, num_bins; growth_cutoff)
        for i in 1:num_bins
            si = round(Int, (i - 1)*length(df[!, :norm_time]) / num_bins) + 1
            ei = min(round(Int, i*length(df[!, :norm_time]) / num_bins) + 1, length(df[!, :norm_time]))

            fit = fits[i]

            if fit == [0.0, 0.0]
                continue
            end

            xs = [df[si, :norm_time], df[ei, :norm_time]]
            ys = exp_model(xs, fit)

            scatterlines!(ax, xs ./ (2pi), ys, color=:red, linewidth=1)
        end
    end

    hlines!(ax, [10.0^growth_cutoff], color=:black, linestyle=:dash)
end

function make_fit_plot2(df; num_bins=10, show_fits=false)
    fig = Figure(size=(325, 250), fonts=(; regular="Times New Roman"), fontsize=14)

    ax1 = Axis(fig[1, 1], yscale=log10)
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.01.csv", DataFrame)
    make_combo_fit_plot_axis(ax1, df; show_fits)

    ax1.xlabel = L"t \omega_p / 2 \pi"
    ax1.limits = ((0, 10), nothing)
    save("fit.pdf", fig)
end

function make_combo_fit_plot(; num_bins=10)
    fig = Figure(size=(325, 250), fonts=(; regular="Times New Roman"), fontsize=14)

    @info "First"
    ax1 = Axis(fig[1, 1], yscale=log10)
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.01.csv", DataFrame)
    make_combo_fit_plot_axis(ax1, df)
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false
    ax1.limits = ((0, 10), (1e-6, 1e2))

    @info "Second"
    ax2 = Axis(fig[2, 1], yscale=log10)
    # ax2 = Axis(fig[2, 1]) 
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.1.csv", DataFrame)
    make_combo_fit_plot_axis(ax2, df)
    ax2.xlabel = L"t \omega_p / 2 \pi"
    ax2.limits = ((0, 10), (1e-6, 1e2))

    save("combo_fit.pdf", fig)
end

df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.1.csv", DataFrame)
# make_fit_plot(df, show_fits=true)
# make_fit_plot2(df, show_fits=false)

make_combo_fit_plot()

function compute_max_growth_rate(df; num_bins=10)
    fits = compute_fits(df, num_bins)
    if num_bins == 1
        return fits[1][1]
    end
    return max(maximum(x -> x[1], fits[2:end]), 0)
end

function compute_growth_rates(algo)
    norm_beam_vels = collect(range(0.0, 0.4, step=0.01))
    norm_therm_vels = collect(range(0.01, 0.35, step=0.01))

    growth_rates = Matrix{Float64}(undef, length(norm_therm_vels), length(norm_beam_vels))

    for (i, norm_therm_vel) = enumerate(norm_therm_vels), (j, norm_beam_vel) = enumerate(norm_beam_vels)
        try
            df = read_data(algo, norm_beam_vel, norm_therm_vel)
            growth_rate = compute_max_growth_rate(df)

            if growth_rate < 0
                @warn "Negative growth rate" norm_therm_vel norm_beam_vel growth_rate
            end

            growth_rates[i, j] = growth_rate
        catch err
            @error "Error measuring growth rate" algo norm_therm_vel norm_beam_vel err
            growth_rates[i, j] = 0
        end
    end

    return norm_beam_vels, norm_therm_vels, growth_rates
end

function make_growth_axis(ax, algo; hidex=false, hidey=false, colorrange=(-4, 0), v_critical=nothing)
    @info "Making growth axis" algorithm=algo hidex hidey

    norm_beam_vels, norm_therm_vels, growth_rates = compute_growth_rates(algo)

    hm = heatmap!(ax, norm_beam_vels, norm_therm_vels, log10.(transpose(growth_rates)); colorrange, colormap=:inferno)

    # ax.xlabel = L"v_b / \omega_p \Delta x"
    # ax.ylabel = L"v_t / \omega_p \Delta x"
    ax.xlabel = L"\bar{v}_d"
    ax.ylabel = L"\bar{v}_t"
    ax.ylabelrotation = 0

    ax.xticks = ([0.1, 0.2, 0.3, 0.4], [L"0.1", L"0.2", L"0.3", L"0.4"])
    ax.yticks = ([0.1, 0.2, 0.3], [L"0.1", L"0.2", L"0.3"])

    if hidex
        ax.xlabelvisible = false
        # ax.xticklabelsvisible = false
        # ax.xticksvisible = false
    end

    if hidey
        ax.ylabelvisible = false
        ax.yticklabelsvisible = false
        ax.yticksvisible = false
    end

    if !isnothing(v_critical)
        ax.xminorticks = [v_critical]
        ax.xminortickcolor = :red
        ax.xminorticksvisible = true
        ax.xminorticksize = 7
        ax.xminortickwidth = 3
    end

    return hm
end

function make_growth_plot(algo; title=algo)

    fig = Figure()

    ax1 = Axis(fig[1,1], aspect=DataAspect(), title=title)
    hm = make_growth_axis(ax1, algo)

    cbar = Colorbar(fig[:, 2], hm, label=L"\text{(Growth rate)} / \omega_p")
    cbar.ticks = ([-3, -2, -1, 0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])

    save("$(algo).pdf", fig)
end

# make_growth_plot("mcpic1")
# make_growth_plot("ecpic1")
# make_growth_plot("ecpic2")
# make_growth_plot("ecpic2_new"; title="ecpic2 w/ 5 pt field solve")
# make_growth_plot("ecpic2_five"; title="ecpic2 w/ 4th order solve")
# make_growth_plot("pics")

function make_combo_growth_heatmap()
    # 624 units corresponds to a width of 6.5 inches
    fig = Figure(size=(624, 400))

    ax1 = Axis(fig[1,1], aspect=DataAspect(), title="MC-PIC1")
    hm = make_growth_axis(ax1, "mcpic1", hidex=true)

    ax2 = Axis(fig[1,2], aspect=DataAspect(), title="EC-PIC1")
    make_growth_axis(ax2, "ecpic1", hidex=true, hidey=true, v_critical=0.288)

    ax3 = Axis(fig[1,3], aspect=DataAspect(), title="EC-PIC2-Standard")
    make_growth_axis(ax3, "ecpic2", hidex=true, hidey=true, v_critical=0.183)

    ax4 = Axis(fig[2,1], aspect=DataAspect(), title="EC-PIC2-Fourth")
    hm = make_growth_axis(ax4, "ecpic2_five"; v_critical=0.158)

    ax5 = Axis(fig[2,2], aspect=DataAspect(), title="EC-PIC2-Lagrange")
    make_growth_axis(ax5, "ecpic2_new", hidey=true, v_critical=0.316)

    ax6 = Axis(fig[2,3], aspect=DataAspect(), title="CS-PIC")
    make_growth_axis(ax6, "pics", hidey=true)

    cbar = Colorbar(fig[:, 4], hm, label=L"\text{(Growth rate)} / \omega_p")
    # cbar.ticks = ([-3, -2, -1, 0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
    # cbar.ticks = ([-5, -4, -3, -2, -1, 0], [L"10^{-5}",L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
    cbar.ticks = ([-4, -2, 0], [L"10^{-4}", L"10^{-2}", L"10^{0}"])

    save("combo_growth_heatmap.pdf", fig)
end

# make_combo_growth_heatmap()

function make_combo_growth_heatmap2()
    # 624 units corresponds to a width of 6.5 inches
    fig = Figure(size=(624, 300))

    ax1 = Axis(fig[1,1], aspect=DataAspect(), title="MC-PIC1")
    hm = make_growth_axis(ax1, "mcpic1", hidex=true)

    ax2 = Axis(fig[1,2], aspect=DataAspect(), title="EC-PIC1")
    make_growth_axis(ax2, "ecpic1", hidex=true, hidey=true, v_critical=0.288)

    ax3 = Axis(fig[1,3], aspect=DataAspect(), title="CS-PIC")
    make_growth_axis(ax3, "pics", hidex=true, hidey=true)

    ax4 = Axis(fig[2,1], aspect=DataAspect(), title="EC-PIC2-Fourth")
    hm = make_growth_axis(ax4, "ecpic2_five"; v_critical=0.158)

    ax5 = Axis(fig[2,2], aspect=DataAspect(), title="EC-PIC2-Lagrange")
    make_growth_axis(ax5, "ecpic2_new", hidey=true, v_critical=0.316)

    ax6 = Axis(fig[2,3], aspect=DataAspect(), title="EC-PIC2-Standard")
    make_growth_axis(ax6, "ecpic2", hidey=true, v_critical=0.183)

    cbar = Colorbar(fig[:, 4], hm, label=L"\text{(Growth rate)} / \omega_p")
    # cbar.ticks = ([-3, -2, -1, 0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
    # cbar.ticks = ([-5, -4, -3, -2, -1, 0], [L"10^{-5}",L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
    cbar.ticks = ([-4, -2, 0], [L"10^{-4}", L"10^{-2}", L"10^{0}"])

    save("combo_growth_heatmap2.pdf", fig)
end

# make_combo_growth_heatmap2()

function make_timestep_plot(algo; title=algo, scale_line=false)
    @info "Making timestep plot" algo title scale_line
    fig = Figure()

    ax1 = Axis(fig[1,1], title=title, xscale=log10, yscale=log10)
    ax1.xlabel = "Normalized timestep"
    ax1.ylabel = "Relative error in plasma freq"
    # ax1.limits = (nothing, (1e-3, 1))

    norm_timesteps = [1.0, 0.1, 0.01, 0.001, 0.0001]
    measured_plasma_freqs = Float64[]
    measured_errors = Float64[]
    for nt in norm_timesteps
        # df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)
        df = CSV.read("data/algo=$(algo)_nt=$(nt).csv", DataFrame)

        exact_plasma_freq = 5.685352436149611e8
        measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :field_energy])

        push!(measured_plasma_freqs, measured_freq)
        push!(measured_errors, abs(measured_freq / exact_plasma_freq - 1))
    end

    scatter!(ax1, norm_timesteps, measured_errors)

    if scale_line
        p = 1
        lines!(ax1, norm_timesteps, norm_timesteps.^p / norm_timesteps[1]^p * measured_errors[1], color=:black, linestyle=:dash)
        p = 2
        lines!(ax1, norm_timesteps, norm_timesteps.^p / norm_timesteps[1]^p * measured_errors[1], color=:black)
    end

    save("$(algo)_timestep.pdf", fig)
end
# make_timestep_plot("ecpic2_five", scale_line=true)
# make_timestep_plot("mcpic1", scale_line=true)


function compute_stationary_growth_rates(algo, ppc; norm_therm_vels = collect(range(0.01, 0.3, step=0.01)), num_bins=10)
    growth_rates = Vector{Float64}(undef, length(norm_therm_vels))

    for (i, norm_therm_vel) = enumerate(norm_therm_vels)
        try
            df = CSV.read("data/algo=$(algo)_bm=0.0_tm=$(norm_therm_vel)_ppc=$(ppc).csv", DataFrame)
            growth_rate = compute_max_growth_rate(df; num_bins)

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

function stationary_stab_plot(algo)
    fig = Figure(size=(325, 300), fonts=(; regular="Times New Roman"), fontsize=14)

    ax1 = Axis(fig[1, 1], yscale=log10)

    for ppc in [1000, 10000, 100000, 1000000]
        norm_therm_vels, growth_rates = compute_stationary_growth_rates(algo, ppc)
        ppc_exp = round(Int, log10(ppc))
        lines!(ax1, norm_therm_vels, growth_rates, label=L"ppc=10^%$(ppc_exp)")
    end

    norm_therm_vels, growth_rates = compute_stationary_growth_rates(algo, 10000000, norm_therm_vels=[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2], num_bins=1)
    lines!(ax1, norm_therm_vels, growth_rates, label=L"ppc=10^7")

    ax1.xlabel = L"\bar{v}_t"
    ax1.ylabel = L"\gamma / \omega_p"

    axislegend(ax1)

    save("stationary_$(algo).pdf", fig)
end
# stationary_stab_plot("mcpic1")

# df = CSV.read("data/algo=mcpic1_bm=0.0_tm=0.1_ppc=1000000.csv", DataFrame)
# make_fit_plot(df, show_fits=true)
