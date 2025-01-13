using CSV
using DataFrames
using LsqFit
using FFTW
using CairoMakie
using Colors
using ColorSchemes

function read_data(algo, bm, tm)
    CSV.read("data/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
    # CSV.read("data20240419/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
end

function compute_fit2(df; timespan=10, growth_cutoff=-2)
    ynorm = df[1, :thermal_energy]
    norm_energy = (df[!, :thermal_energy] .- ynorm) ./ ynorm

    imax = findfirst(x -> x > 10.0^growth_cutoff, norm_energy)
    if imax === nothing
        imax = length(norm_energy)
    end

    tmin = df[imax, :norm_time] - 2*pi*timespan
    imin = findfirst(x -> x > tmin, df[!, :norm_time])
    if imin === nothing
        imin = 1
    end

    xs = df[imin:imax, :norm_time]
    ys = norm_energy[imin:imax]

    @. model(x, p) = (p[1] * x) + p[2]

    log_ys = Vector{Float64}()
    new_xs = Vector{Float64}()
    for i in eachindex(ys)
        log_y = sign(ys[i]) * log(abs(ys[i]))
        if !(isnan(log_y) || isinf(log_y))
            push!(new_xs, xs[i])
            push!(log_ys, log_y)
        end
    end

    fit = curve_fit(model, new_xs, log_ys, [1e-12, log_ys[1]])

    return fit
end

function compute_fit(xs, ys; growth_cutoff=-3)
    @. model(x, p) = (p[1] * x) + p[2]

    # if any(ys .< -1 * 10.0^growth_cutoff)
    #     return [0.0, 0.0]
    # end

    log_ys = Vector{Float64}()
    new_xs = Vector{Float64}()
    for i in eachindex(ys)
        log_y = sign(ys[i]) * log(abs(ys[i]))
        # log_y = log(abs(ys[i]))
        if !(isnan(log_y) || isinf(log_y))
            push!(new_xs, xs[i])
            push!(log_ys, log_y)
        end
    end

    fit = curve_fit(model, new_xs, log_ys, [1e-12, log_ys[1]])

    # if margin_error(fit)[1] > 0.1 && xs[end] > 600
    #     return [0.0, 0.0]
    # end

    return coef(fit)
end

function compute_fits(df, num_bins=20; growth_cutoff=-3)
    fits = Vector{Vector{Float64}}(undef, num_bins)

    # if any((df[:, :thermal_energy] .- df[1, :thermal_energy]) ./ df[1, :thermal_energy] .< -1 * 10.0^growth_cutoff)
    #     for i in 1:num_bins
    #         fits[i] = [0.0, 0.0]
    #     end
    #     return fits
    # end

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

function make_combo_fit_plot_axis(ax, df; num_bins=10, show_fits=true, growth_cutoff=-2, change_in_energy=true, show_negative=false, lower_ppc=nothing)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.ylabel = L"|\Delta E_\text{th}| / E_\text{th}(0)"

    ynorm = df[1, :thermal_energy]

    if show_negative
        norm_energy = (df[!, :thermal_energy] .- ynorm) ./ ynorm
        color_val = sign.(norm_energy)
        cs = ColorScheme([colorant"blue", colorant"black"])
        lines!(ax, df[!, :norm_time] ./ (2pi), abs.(norm_energy), color=color_val, colormap=cs, linewidth=2)

        if lower_ppc !== nothing
            ynorm2 = lower_ppc[1, :thermal_energy]
            norm_energy2 = (lower_ppc[!, :thermal_energy] .- ynorm2) ./ ynorm2
            color_val2 = sign.(norm_energy2)
            cs2 = ColorScheme([alphacolor(colorant"blue", 0.5), alphacolor(colorant"black", 0.5)])
            lines!(ax, lower_ppc[!, :norm_time] ./ (2pi), abs.(norm_energy2), color=color_val2, colormap=cs2, linewidth=2)
        end
    elseif change_in_energy
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

            scatterlines!(ax, xs ./ (2pi), ys, color=:red, linewidth=0.5, markersize=2)
        end
    end

    # ax.title = L"\gamma / \omega_p = %$(max_growth)"
    hlines!(ax, [10.0^growth_cutoff], color=:black, linestyle=:dash)
end

function make_combo_fit_plot_axis2(ax, df; num_bins=10, show_fits=true, growth_cutoff=-2, change_in_energy=true, show_negative=true, lower_ppc=nothing)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.ylabel = L"|\Delta E_\text{th}| / E_\text{th}(0)"

    ynorm = df[1, :thermal_energy]

    @assert show_negative
    @assert change_in_energy

    norm_energy = (df[!, :thermal_energy] .- ynorm) ./ ynorm
    color_val = sign.(norm_energy)
    cs = ColorScheme([colorant"blue", colorant"black"])
    lines!(ax, df[!, :norm_time] ./ (2pi), abs.(norm_energy), color=color_val, colormap=cs, linewidth=2)

    if lower_ppc !== nothing
        ynorm2 = lower_ppc[1, :thermal_energy]
        norm_energy2 = (lower_ppc[!, :thermal_energy] .- ynorm2) ./ ynorm2
        color_val2 = sign.(norm_energy2)
        cs2 = ColorScheme([alphacolor(colorant"blue", 0.5), alphacolor(colorant"black", 0.5)])
        lines!(ax, lower_ppc[!, :norm_time] ./ (2pi), abs.(norm_energy2), color=color_val2, colormap=cs2, linewidth=2)
    end

    max_growth = 0.0

    if show_fits
        imax = findfirst(x -> x > 10.0^growth_cutoff, norm_energy)
        if imax === nothing
            imax = length(norm_energy)
        end

        tmin = df[imax, :norm_time] - 2*pi*10
        imin = findfirst(x -> x > tmin, df[!, :norm_time])
        if imin === nothing
            imin = 1
        end

        fit = compute_fit(df[imin:imax, :norm_time], norm_energy[imin:imax]; growth_cutoff=-Inf)

        @show fit

        xs = [df[imin, :norm_time], df[imax, :norm_time]]
        ys = exp_model(xs, fit)

        scatterlines!(ax, xs ./ (2pi), ys, color=:red, linewidth=2, markersize=2)
    end

    # ax.title = L"\gamma / \omega_p = %$(max_growth)"
    hlines!(ax, [10.0^growth_cutoff], color=:black, linestyle=:dash)
end

function make_fit_plot(df; num_bins=100, show_fits=false, change_in_energy=true, range=(-4, 2), growth_cutoff=-2, show_negative=false, filename="fit.pdf", lower_ppc=nothing)
    fig = Figure(size=(325, 220), fonts=(; regular="Times New Roman"), fontsize=14)

    ax1 = Axis(fig[1, 1], yscale=log10)
    make_combo_fit_plot_axis2(ax1, df; show_fits, num_bins, change_in_energy, growth_cutoff, show_negative, lower_ppc)

    ax1.xlabel = L"t \omega_p / 2 \pi"
    ax1.limits = (nothing, (10.).^(range))
    save(filename, fig)
end

function make_combo_fit_plot(; num_bins=100, growth_cutoff=-2)
    fig = Figure(size=(325, 250), fonts=(; regular="Times New Roman"), fontsize=14)

    @info "First"
    ax1 = Axis(fig[1, 1], yscale=log10)
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.01.csv", DataFrame)
    make_combo_fit_plot_axis(ax1, df; num_bins, growth_cutoff)
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false
    ax1.limits = ((0, 100), (1e-6, 1e2))

    @info "Second"
    ax2 = Axis(fig[2, 1], yscale=log10)
    # ax2 = Axis(fig[2, 1]) 
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.1.csv", DataFrame)
    make_combo_fit_plot_axis(ax2, df; num_bins, growth_cutoff)
    ax2.xlabel = L"t \omega_p / 2 \pi"
    ax2.limits = ((0, 100), (1e-6, 1e2))

    save("combo_fit.pdf", fig)
end

function compute_max_growth_rate(df; num_bins=10, growth_cutoff=-2)
    ynorm = df[1, :thermal_energy]
    norm_energy = (df[!, :thermal_energy] .- ynorm) ./ ynorm

    imax = findfirst(x -> x > 10.0^growth_cutoff, norm_energy)
    if imax === nothing
        imax = length(norm_energy)
    end

    tmin = df[imax, :norm_time] - 10*2*pi
    imin = findfirst(x -> x > tmin, df[!, :norm_time])
    if imin === nothing
        imin = 1
    end

    fit = compute_fit(df[imin:imax, :norm_time], norm_energy[imin:imax]; growth_cutoff=Inf)
    # fit = compute_fit(df[imin:imax, :norm_time], norm_energy[imin:imax]; growth_cutoff=-10)

    return max(fit[1], 0) / 2
end

function compute_growth_rates(algo; growth_cutoff=-2, num_bins=100)
    # norm_beam_vels = collect(range(0.0, 0.5, step=0.01))
    # norm_therm_vels = collect(range(0.01, 0.35, step=0.01))
    norm_beam_vels = collect(range(0.0, 0.45, step=0.01))
    norm_therm_vels = collect(range(0.01, 0.25, step=0.01))

    growth_rates = Matrix{Float64}(undef, length(norm_therm_vels), length(norm_beam_vels))

    for (i, norm_therm_vel) = enumerate(norm_therm_vels), (j, norm_beam_vel) = enumerate(norm_beam_vels)
        try
            df = read_data(algo, norm_beam_vel, norm_therm_vel)
            growth_rate = compute_max_growth_rate(df; growth_cutoff, num_bins)

            if growth_rate < 0
                @warn "Negative growth rate" norm_therm_vel norm_beam_vel growth_rate
            end

            growth_rates[i, j] = growth_rate
        catch err
            # @error "Error measuring growth rate" algo norm_therm_vel norm_beam_vel err
            growth_rates[i, j] = +Inf
        end
    end

    return norm_beam_vels, norm_therm_vels, growth_rates
end

function make_growth_axis(ax, algo; hidex=false, hidey=false, colorrange=(-3, 0), v_critical=nothing, growth_cutoff=-2, num_bins=100)
    @info "Making growth axis" algorithm=algo hidex hidey

    norm_beam_vels, norm_therm_vels, growth_rates = compute_growth_rates(algo; growth_cutoff, num_bins)

    hm = heatmap!(ax, norm_beam_vels, norm_therm_vels, log10.(transpose(growth_rates)); colorrange, colormap=:inferno, lowclip=:black, highclip=:white)
    # hm = heatmap!(ax, norm_beam_vels, norm_therm_vels, log10.(transpose(growth_rates)); colorrange, colormap=:inferno, lowclip=:white)

    # ax.xlabel = L"v_b / \omega_p \Delta x"
    # ax.ylabel = L"v_t / \omega_p \Delta x"
    ax.xlabel = L"\bar{v}_d"
    ax.ylabel = L"\bar{v}_t"
    ax.ylabelrotation = 0

    ax.xticks = ([0.1, 0.2, 0.3, 0.4, 0.5], [L"0.1", L"0.2", L"0.3", L"0.4", L"0.5"])
    ax.yticks = ([0.0, 0.1, 0.2, 0.3], [L"0.0", L"0.1", L"0.2", L"0.3"])
    # ax.limits = ((0, 0.5), (0.01, 0.35))
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

function make_growth_plot(algo; title=algo, growth_cutoff=-2)

    fig = Figure()

    ax1 = Axis(fig[1,1], aspect=DataAspect(), title=title)
    hm = make_growth_axis(ax1, algo, growth_cutoff=growth_cutoff)

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

function make_combo_growth_heatmap(; growth_cutoff=-2, num_bins=20)
    # 624 units corresponds to a width of 6.5 inches
    # fig = Figure(size=(624, 400))
    fig = Figure(size=(640, 400))

    growth_cutoff = -12
    ax1 = Axis(fig[1,1], aspect=DataAspect(), title="MC-PIC1 ($growth_cutoff)")
    hm = make_growth_axis(ax1, "mcpic1", hidex=true, growth_cutoff=growth_cutoff, num_bins=num_bins)

    growth_cutoff = -5
    ax2 = Axis(fig[1,2], aspect=DataAspect(), title="EC-PIC1 ($growth_cutoff)")
    make_growth_axis(ax2, "ecpic1", hidex=true, hidey=true, v_critical=0.288, growth_cutoff=growth_cutoff, num_bins=num_bins)

    growth_cutoff = -10
    ax3 = Axis(fig[1,3], aspect=DataAspect(), title="EC-PIC2-Standard ($growth_cutoff)")
    make_growth_axis(ax3, "ecpic2", hidex=true, hidey=true, v_critical=0.183, growth_cutoff=growth_cutoff, num_bins=num_bins)

    growth_cutoff = -10
    ax4 = Axis(fig[2,1], aspect=DataAspect(), title="EC-PIC2-Fourth ($growth_cutoff)")
    hm = make_growth_axis(ax4, "ecpic2_five"; v_critical=0.158, growth_cutoff=growth_cutoff, num_bins=num_bins)

    growth_cutoff = -8
    ax5 = Axis(fig[2,2], aspect=DataAspect(), title="EC-PIC2-Lagrange ($growth_cutoff)")
    make_growth_axis(ax5, "ecpic2_new", hidey=true, v_critical=0.316, growth_cutoff=growth_cutoff, num_bins=num_bins)

    growth_cutoff = -5
    ax6 = Axis(fig[2,3], aspect=DataAspect(), title="CS-PIC ($growth_cutoff)")
    make_growth_axis(ax6, "pics", hidey=true, growth_cutoff=growth_cutoff, num_bins=num_bins)

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

growth_cutoff = -2
# make_combo_fit_plot(; growth_cutoff)
# make_combo_growth_heatmap(; growth_cutoff, num_bins=100)
# stationary_stab_plot("mcpic1"; growth_cutoff)

# df = CSV.read("data/algo=ecpic1_bm=0.1_tm=0.05.csv", DataFrame)
# df = CSV.read("data/algo=ecpic1_bm=0.1_tm=0.2.csv", DataFrame)
# df = CSV.read("data/algo=ecpic1_bm=0.3_tm=0.22.csv", DataFrame)
# df = CSV.read("data/algo=mcpic1_bm=0.0_tm=0.07_ppc=$(2^20).csv", DataFrame)
# df = CSV.read("data/algo=mcpic1_bm=0.0_tm=0.02_ppc=$(2^16)_init_strat=quiet.csv", DataFrame)

# df = CSV.read("data/algo=ecpic1_bm=0.3_tm=0.22.csv", DataFrame)
# df = CSV.read("data/algo=ecpic1_bm=0.3_tm=0.22.csv", DataFrame)

# df = CSV.read("data/algo=ecpic1_bm=0.2_tm=0.07.csv", DataFrame)

df = CSV.read("data/algo=mcpic1_bm=0.1_tm=0.02.csv", DataFrame)
df2 = CSV.read("data/algo=mcpic1_ppc=$(2^14)_bm=0.1_tm=0.02.csv", DataFrame)

# df = CSV.read("data/algo=ecpic1_bm=0.1_tm=0.11.csv", DataFrame)
# df2 = CSV.read("data/algo=ecpic1_ppc=$(2^14)_bm=0.1_tm=0.11.csv", DataFrame)
make_fit_plot(df, show_fits=true, growth_cutoff=-2, range=(-15,2), show_negative=true, num_bins=20, lower_ppc=df2)

