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

    if log10(abs(ys[1])) < growth_cutoff
        # @info "Growth rate too small" coef(fit) log10(ys[end]) log_ys[end]
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

function make_combo_fit_plot_axis(ax, df; num_bins=10, show_fits=true, growth_cutoff=-2, change_in_energy=true)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.ylabel = L"|\Delta E_\text{th}| / E_\text{th}(0)"

    ynorm = df[1, :thermal_energy]

    if change_in_energy
        lines!(ax, df[!, :norm_time] ./ (2pi), abs.(df[!, :thermal_energy] .- ynorm) ./ ynorm, color=:black, linewidth=2)
    else
        lines!(ax, df[!, :norm_time] ./ (2pi), abs.(df[!, :thermal_energy]), color=:black, linewidth=2)
    end

    max_growth = 0.0

    if show_fits
        fits = compute_fits(df, num_bins; growth_cutoff)
        for i in 1:num_bins
            si = round(Int, (i - 1)*length(df[!, :norm_time]) / num_bins) + 1
            ei = min(round(Int, i*length(df[!, :norm_time]) / num_bins) + 1, length(df[!, :norm_time]))

            fit = fits[i]

            if fit == [0.0, 0.0]
                continue
            end

            if fit[1] > max_growth
                max_growth = fit[1]
            end

            xs = [df[si, :norm_time], df[ei, :norm_time]]
            ys = exp_model(xs, fit)

            scatterlines!(ax, xs ./ (2pi), ys, color=:red, linewidth=1)
        end
    end

    # ax.title = L"\gamma / \omega_p = %$(max_growth)"
    hlines!(ax, [10.0^growth_cutoff], color=:black, linestyle=:dash)
end

function make_fit_plot(df; num_bins=10, show_fits=false)
    fig = Figure(size=(325, 250), fonts=(; regular="Times New Roman"), fontsize=14)

    ax1 = Axis(fig[1, 1], yscale=log10)
    make_combo_fit_plot_axis(ax1, df; show_fits)

    ax1.xlabel = L"t \omega_p / 2 \pi"
    # ax1.limits = ((0, 10), nothing)
    save("fit.pdf", fig)
end

function make_combo_fit_plot(; num_bins=10, growth_cutoff=-2)
    fig = Figure(size=(325, 250), fonts=(; regular="Times New Roman"), fontsize=14)

    @info "First"
    ax1 = Axis(fig[1, 1], yscale=log10)
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.01.csv", DataFrame)
    make_combo_fit_plot_axis(ax1, df; num_bins, growth_cutoff)
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false
    ax1.limits = ((0, 10), (1e-6, 1e2))

    @info "Second"
    ax2 = Axis(fig[2, 1], yscale=log10)
    # ax2 = Axis(fig[2, 1]) 
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.1.csv", DataFrame)
    make_combo_fit_plot_axis(ax2, df; num_bins, growth_cutoff)
    ax2.xlabel = L"t \omega_p / 2 \pi"
    ax2.limits = ((0, 10), (1e-6, 1e2))

    save("combo_fit.pdf", fig)
end

function compute_max_growth_rate(df; num_bins=10, growth_cutoff=-2)
    fits = compute_fits(df, num_bins; growth_cutoff)
    if num_bins == 1
        return fits[1][1]
    end
    return max(maximum(x -> x[1], fits[2:end]), 0)
end

function compute_growth_rates(algo; growth_cutoff=-2)
    norm_beam_vels = collect(range(0.0, 0.5, step=0.01))
    norm_therm_vels = collect(range(0.01, 0.35, step=0.01))

    growth_rates = Matrix{Float64}(undef, length(norm_therm_vels), length(norm_beam_vels))

    for (i, norm_therm_vel) = enumerate(norm_therm_vels), (j, norm_beam_vel) = enumerate(norm_beam_vels)
        try
            df = read_data(algo, norm_beam_vel, norm_therm_vel)
            growth_rate = compute_max_growth_rate(df; growth_cutoff)

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

function make_growth_axis(ax, algo; hidex=false, hidey=false, colorrange=(-3, 0), v_critical=nothing, growth_cutoff=-2)
    @info "Making growth axis" algorithm=algo hidex hidey

    norm_beam_vels, norm_therm_vels, growth_rates = compute_growth_rates(algo; growth_cutoff)

    hm = heatmap!(ax, norm_beam_vels, norm_therm_vels, log10.(transpose(growth_rates)); colorrange, colormap=:inferno, lowclip=:black)
    # hm = heatmap!(ax, norm_beam_vels, norm_therm_vels, log10.(transpose(growth_rates)); colorrange, colormap=:inferno, lowclip=:white)

    # ax.xlabel = L"v_b / \omega_p \Delta x"
    # ax.ylabel = L"v_t / \omega_p \Delta x"
    ax.xlabel = L"\bar{v}_d"
    ax.ylabel = L"\bar{v}_t"
    ax.ylabelrotation = 0

    ax.xticks = ([0.1, 0.2, 0.3, 0.4, 0.5], [L"0.1", L"0.2", L"0.3", L"0.4", L"0.5"])
    ax.yticks = ([0.1, 0.2, 0.3], [L"0.1", L"0.2", L"0.3"])
    ax.limits = ((0, 0.5), (0.01, 0.35))

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

function make_combo_growth_heatmap(; growth_cutoff=-2)
    # 624 units corresponds to a width of 6.5 inches
    # fig = Figure(size=(624, 400))
    fig = Figure(size=(624, 340))

    ax1 = Axis(fig[1,1], aspect=DataAspect(), title="MC-PIC1")
    hm = make_growth_axis(ax1, "mcpic1", hidex=true, growth_cutoff=growth_cutoff)

    ax2 = Axis(fig[1,2], aspect=DataAspect(), title="EC-PIC1")
    make_growth_axis(ax2, "ecpic1", hidex=true, hidey=true, v_critical=0.288, growth_cutoff=growth_cutoff)

    ax3 = Axis(fig[1,3], aspect=DataAspect(), title="EC-PIC2-Standard")
    make_growth_axis(ax3, "ecpic2", hidex=true, hidey=true, v_critical=0.183, growth_cutoff=growth_cutoff)

    ax4 = Axis(fig[2,1], aspect=DataAspect(), title="EC-PIC2-Fourth")
    hm = make_growth_axis(ax4, "ecpic2_five"; v_critical=0.158, growth_cutoff=growth_cutoff)

    ax5 = Axis(fig[2,2], aspect=DataAspect(), title="EC-PIC2-Lagrange")
    make_growth_axis(ax5, "ecpic2_new", hidey=true, v_critical=0.316, growth_cutoff=growth_cutoff)

    ax6 = Axis(fig[2,3], aspect=DataAspect(), title="CS-PIC")
    make_growth_axis(ax6, "pics", hidey=true, growth_cutoff=growth_cutoff)

    cbar = Colorbar(fig[:, 4], hm, label=L"\text{(Growth rate)} / \omega_p")
    # cbar.ticks = ([-3, -2, -1, 0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
    # cbar.ticks = ([-5, -4, -3, -2, -1, 0], [L"10^{-5}",L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
    cbar.ticks = ([-3, -2, -1, 0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])

    save("combo_growth_heatmap.pdf", fig)
end

function compute_stationary_growth_rates(algo, ppc; norm_therm_vels = collect(range(0.01, 0.3, step=0.01)), num_bins=10, growth_cutoff=-2)
    growth_rates = Vector{Float64}(undef, length(norm_therm_vels))

    for (i, norm_therm_vel) = enumerate(norm_therm_vels)
        try
            # df = CSV.read("data/algo=$(algo)_bm=0.0_tm=$(norm_therm_vel)_ppc=$(ppc).csv", DataFrame)
            df = CSV.read("data20241213/algo=$(algo)_bm=0.0_tm=$(norm_therm_vel)_ppc=$(ppc).csv", DataFrame)
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

    ppcs = [1000, 10000, 100000, 1000000]
    limits = [-1, -2, -3, -4]

    for (ppc, limit) in zip(ppcs, limits)
        norm_therm_vels, growth_rates = compute_stationary_growth_rates(algo, ppc; growth_cutoff=limit)
        # growth_rates = [g == 0 ? 1e-9 : g for g in growth_rates]
        ppc_exp = round(Int, log10(ppc))
        lines!(ax1, norm_therm_vels, growth_rates, label=L"ppc=10^%$(ppc_exp)")
    end

    ax1.xlabel = L"\bar{v}_t"
    ax1.ylabel = L"\gamma / \omega_p"

    ax1.limits = ((0.0, 0.3), (1e-3, 1e-1))

    axislegend(ax1)

    save("stationary_$(algo).pdf", fig)
end

growth_cutoff = -2
make_combo_fit_plot(; growth_cutoff)
make_combo_growth_heatmap(; growth_cutoff)
stationary_stab_plot("mcpic1"; growth_cutoff)

# df = CSV.read("data/algo=pics_bm=0.4_tm=0.3.csv", DataFrame)
# df = CSV.read("data/algo=pics_bm=0.4_tm=0.01.csv", DataFrame)
# make_fit_plot(df, show_fits=true)
