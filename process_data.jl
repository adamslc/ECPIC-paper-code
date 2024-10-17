using CSV
using DataFrames
using LsqFit
using FFTW
using CairoMakie

function read_data(algo, bm, tm)
    CSV.read("data/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
    # CSV.read("data20240419/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
end

function compute_fit(xs, ys)
    @. model(x, p) = (p[1] * x) + p[2]

    log_ys = Vector{Float64}()
    new_xs = Vector{Float64}()
    for i in eachindex(ys)
        log_y = log(ys[i])
        if !(isnan(log_y) || isinf(log_y))
            push!(new_xs, xs[i])
            push!(log_ys, log_y)
        end
    end

    fit = curve_fit(model, new_xs, log_ys, [1e-12, log_ys[1]])

    return coef(fit)
end

function compute_fits(df, num_bins=20)
    fits = Vector{Vector{Float64}}(undef, num_bins)

    for i in 1:num_bins
        si = round(Int, (i - 1)*length(df[!, :norm_time]) / num_bins) + 1
        ei = min(round(Int, i*length(df[!, :norm_time]) / num_bins) + 1, length(df[!, :norm_time]))

        fit = compute_fit(df[si:ei, :norm_time], df[si:ei, :thermal_energy])

        fits[i] = fit
    end

    return fits
end

@. exp_model(x, p) = exp((p[1] * x) + p[2])

function make_fit_plot(df; num_bins=10, show_fits=false)
    # fig, ax, plt = lines(df[!, :norm_time], df[!, :thermal_energy], color=:blue, yscale=:log10)
    fig, ax, plt = lines(df[!, :norm_time], df[!, :thermal_energy], color=:blue)

    ax.yscale = log10

    if show_fits
    fits = compute_fits(df, num_bins)
        for i in 1:num_bins
            si = round(Int, (i - 1)*length(df[!, :norm_time]) / num_bins) + 1
            ei = min(round(Int, i*length(df[!, :norm_time]) / num_bins) + 1, length(df[!, :norm_time]))

            fit = fits[i]

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

function make_combo_fit_plot_axis(ax, df; num_bins=10, show_fits=true)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.ylabel = L"E_\text{th} / E_\text{th}(0)"

    ynorm = df[1, :thermal_energy]

    lines!(ax, df[!, :norm_time] ./ (2pi), df[!, :thermal_energy] ./ ynorm, color=:black, linewidth=2)

    if show_fits
        fits = compute_fits(df, num_bins)
        for i in 1:num_bins
            si = round(Int, (i - 1)*length(df[!, :norm_time]) / num_bins) + 1
            ei = min(round(Int, i*length(df[!, :norm_time]) / num_bins) + 1, length(df[!, :norm_time]))

            fit = fits[i]

            xs = [df[si, :norm_time], df[ei, :norm_time]]
            ys = exp_model(xs, fit)

            scatterlines!(ax, xs ./ (2pi), ys ./ ynorm, color=:red, linewidth=1)
            # scatter!(ax, xs ./ (2pi), ys ./ ynorm, color=:red, linewidth=1)
        end
    end
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

    ax1 = Axis(fig[1, 1], yscale=log10)
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.01.csv", DataFrame)
    make_combo_fit_plot_axis(ax1, df)
    ax1.xticklabelsvisible = false
    ax1.xticksvisible = false
    ax1.limits = ((0, 10), (0.9, 100))

    ax2 = Axis(fig[2, 1], yscale=log10)
    # ax2 = Axis(fig[2, 1]) 
    df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.1.csv", DataFrame)
    make_combo_fit_plot_axis(ax2, df)
    ax2.xlabel = L"t \omega_p / 2 \pi"
    ax2.limits = ((0, 10), nothing)

    save("combo_fit.pdf", fig)
end

df = CSV.read("data/algo=ecpic1_bm=0.05_tm=0.01.csv", DataFrame)
# make_fit_plot(df, show_fits=true)
make_fit_plot2(df, show_fits=false)

make_combo_fit_plot()

function compute_max_growth_rate(df; num_bins=10)
    return max(maximum(x -> x[1], compute_fits(df, num_bins)), 0)
end

function compute_growth_rates(algo)
    norm_beam_vels = collect(range(0.0, 0.4, step=0.01))
    norm_therm_vels = collect(range(0.01, 0.2, step=0.01))

    growth_rates = Matrix{Float64}(undef, length(norm_therm_vels), length(norm_beam_vels))

    for (i, norm_therm_vel) = enumerate(norm_therm_vels), (j, norm_beam_vel) = enumerate(norm_beam_vels)
        df = read_data(algo, norm_beam_vel, norm_therm_vel)
        growth_rate = compute_max_growth_rate(df)

        if growth_rate < 0
            @warn "Negative growth rate" norm_therm_vel norm_beam_vel growth_rate
        end

        growth_rates[i, j] = growth_rate
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
    ax.yticks = ([0.1, 0.2], [L"0.1", L"0.2"])

    if hidex
        ax.xlabelvisible = false
        ax.xticklabelsvisible = false
        ax.xticksvisible = false
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
    fig = Figure(size=(624, 300))

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

function make_freq_fit_plot_fft(df)
    fig = Figure()

    ax1 = Axis(fig[1,1])

    n_steps = length(df[!, :norm_time])
    exact_plasma_freq = 5.685352436149611e8
    dt = (df[2, :norm_time] - df[1, :norm_time]) / exact_plasma_freq

    freqs = fftfreq(n_steps, 1 / dt) .* 2pi
    freq_amps = abs.(fft(df[!, :field_energy]))

    freq_amps .= ifelse.(freqs .< 6e8, 0.0, freq_amps)
    max_index = findmax(freq_amps)[2]
    max_freq = freqs[max_index]

    n = min(200, length(freqs))
    lines!(ax1, freqs[1:n], freq_amps[1:n])
    vlines!(ax1, [max_freq], color=:red)
    vlines!(ax1, [exact_plasma_freq * 2], color=:green)

    save("freq_fit_fft.pdf", fig)
end

function measure_plasma_freq_fft(norm_times, energies; exact_plasma_freq = 5.685352436149611e8)
        n_steps = length(norm_times)
        dt = (norm_times[2] - norm_times[1]) / exact_plasma_freq

        freqs = fftfreq(n_steps, 1 / dt) .* 2pi
        freq_amps = abs.(fft(energies))

        freq_amps .= ifelse.(freqs .< 5e8, 0.0, freq_amps)
        max_index = findmax(freq_amps)[2]
        max_freq = freqs[max_index]

        return max_freq / 2
end

# @. freq_model(x, ps) = ps[1] * (1 - cos(x * ps[2])) / 2 + exp(ps[3] * x) - 1
# @. freq_model(x, ps) = ps[1] * (exp(ps[3] * x) - cos(x * ps[2])) / 2
@. freq_model(x, ps) = -1 * ps[1] * exp(ps[3] * x) * sin(x * ps[2])

function make_freq_fit_plot(df)
    fig = Figure()

    ax1 = Axis(fig[1,1])

    times = df[!, :norm_time]
    energies = df[!, :field_energy] / maximum(df[!, :field_energy])
    mode_amps = df[!, :mode_amp] / maximum(df[!, :mode_amp])

    # lines!(ax1, times, energies)
    lines!(ax1, times, df[!, :mode_amp] / maximum(df[!, :mode_amp]))

    exact_plasma_freq = 5.685352436149611e8
    # freq_guess = measure_plasma_freq_fft(times, energies; exact_plasma_freq)
    freq_guess = exact_plasma_freq
    amp_guess = maximum(energies[1:findfirst(x -> x>2*pi, times)])

    # p0s = [1., 2., 0.]
    p0s = [amp_guess, freq_guess / exact_plasma_freq, 0.01]
    fit = curve_fit(freq_model, times, mode_amps, p0s)

    @show coef(fit)

    ts = collect(range(times[1], stop=times[end], length=5000))

    lines!(ax1, ts, freq_model(ts, coef(fit)), linestyle=:dot)
    lines!(ax1, ts, freq_model(ts, p0s), linestyle=:dash)

    save("freq_fit.pdf", fig)
end

function measure_plasma_freq_fit(norm_times, energies; exact_plasma_freq = 5.685352436149611e8)
    new_energies = energies ./ maximum(energies)

    # freq_guess = measure_plasma_freq_fft(norm_times, energies; exact_plasma_freq)
    freq_guess = exact_plasma_freq
    amp_guess = maximum(energies[1:findfirst(x -> x>2*pi, norm_times)])

    # fit = curve_fit(freq_model, norm_times, energies, [amp_guess, 2 * freq_guess / exact_plasma_freq, 0.]; maxIter=10000, x_tol=0, g_tol=0)
    fit = curve_fit(freq_model, norm_times, energies, [amp_guess, freq_guess / exact_plasma_freq, 0.])

    return coef(fit)[2] * exact_plasma_freq
end

# df = CSV.read("data/algo=ecpic2_nw=1.csv", DataFrame)
# df = CSV.read("data/algo=ecpic2_new_nw=8.csv_small_perturb", DataFrame)
# df = CSV.read("data/algo=mcpic1_nt=0.01.csv", DataFrame)
# df = CSV.read("data/algo=ecpic2_new_nw=16_alt.csv", DataFrame)
# make_freq_fit_plot_fft(df)
# make_freq_fit_plot(df)



function make_convergence_plot(algo; title=algo, scale_line=[], long_data=false, ppc_data=false, small_perturb_data=false)
    @info "Making convergence plot" algo title scale_line
    fig = Figure()

    ax1 = Axis(fig[1,1], title=title, xscale=log10, yscale=log10)
    ax1.xlabel = "Wavenumber"
    ax1.ylabel = "Relative error in plasma freq"
    # ax1.limits = (nothing, (1e-6, 1))
    ax1.limits = (nothing, (1e-7, 1))

    norm_wavenumbers = [1, 2, 4, 8, 16, 32, 64]
    measured_plasma_freqs = Float64[]
    measured_errors = Float64[]
    for nw in norm_wavenumbers
        # df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)
        df = CSV.read("data/algo=$(algo)_nw=$(nw).csv", DataFrame)

        exact_plasma_freq = 5.685352436149611e8
        # measured_freq = measure_plasma_freq_fft(df[!, :norm_time], df[!, :field_energy])
        # measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :field_energy])
        measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :mode_amp])

        push!(measured_plasma_freqs, measured_freq)
        push!(measured_errors, abs(measured_freq / exact_plasma_freq - 1))
    end

    # @info "Convergence data" algo measured_plasma_freqs measured_errors

    wavenumbers = norm_wavenumbers ./ 512 .* pi

    scatter!(ax1, wavenumbers, measured_errors, label="Normal")

    for p in scale_line
        lines!(ax1, wavenumbers, wavenumbers.^p / wavenumbers[end]^p * measured_errors[end], color=:black, linestyle=:dash)
    end

    save("$(algo)_convergence.pdf", fig)
end

# make_convergence_plot("mcpic1"; scale_line=[1, 2])
# make_convergence_plot("ecpic1"; scale_line=[1, 2])
# make_convergence_plot("ecpic2"; scale_line=[1, 2])
# make_convergence_plot("ecpic2_new"; scale_line=[1, 2, 3, 4])
# make_convergence_plot("ecpic1_five", scale_line=[1, 2])
# make_convergence_plot("ecpic2_five", scale_line=[1, 2, 3, 4])
# make_convergence_plot("pics", scale_line=[1, 2, 3, 4])


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
