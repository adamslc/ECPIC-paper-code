using CSV
using DataFrames
using LsqFit
using FFTW
using CairoMakie

function read_data(algo, bm, tm)
    # CSV.read("data2/bv=$(bm)_tm=$(tm).csv", DataFrame)
    # CSV.read("data/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
    CSV.read("data20240419/algo=$(algo)_bm=$(bm)_tm=$(tm).csv", DataFrame)
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

function make_fit_plot(df; num_bins=20, show_fits=false)
    fig, ax, plt = lines(df[!, :norm_time], df[!, :thermal_energy], color=:blue)

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

    lines!(ax, df[!, :norm_time], df[!, :field_energy], color=:red)
    lines!(ax, df[!, :norm_time], df[!, :beam_energy], color=:green)
    lines!(ax, df[!, :norm_time], df[!, :beam_energy] + df[!, :thermal_energy] + df[!, :field_energy], color=:black)

    save("fit.pdf", fig)
    # save("fit.png", fig)
end

function compute_max_growth_rate(df; num_bins=20)
    return maximum(x -> x[1], compute_fits(df, num_bins))
end

function compute_growth_rates(algo)
    norm_beam_vels = collect(range(0.01, 0.4, step=0.01))
    norm_therm_vels = collect(range(0.01, 0.2, step=0.01))

    growth_rates = Matrix{Float64}(undef, length(norm_therm_vels), length(norm_beam_vels))

    for (i, norm_therm_vel) = enumerate(norm_therm_vels), (j, norm_beam_vel) = enumerate(norm_beam_vels)
        df = read_data(algo, norm_beam_vel, norm_therm_vel)
        growth_rate = compute_max_growth_rate(df; num_bins=30)

        if growth_rate < 0
            @warn "Negative growth rate" norm_therm_vel norm_beam_vel growth_rate
        end

        growth_rates[i, j] = growth_rate
    end

    return growth_rates
end

function make_growth_plot(algo; title=algo)
    @info algo

    growth_rates = compute_growth_rates(algo)

    fig = Figure()

    ax1 = Axis(fig[1,1], aspect=DataAspect(), title=title)
    hm1 = heatmap!(ax1, log10.(transpose(growth_rates)), colorrange=(-3, 0))

    ax1.xlabel = L"v_b / \omega_p \Delta x"
    ax1.ylabel = L"v_t / \omega_p \Delta x"

    ax1.xticks = ([10, 20, 30, 40], [L"0.1", L"0.2", L"0.3", L"0.4"])
    ax1.yticks = ([10, 20], [L"0.1", L"0.2"])

    cbar = Colorbar(fig[:, 2], hm1, label=L"\text{(Growth rate)} / \omega_p")
    cbar.ticks = ([-3, -2, -1, 0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])

    save("$(algo).pdf", fig)
end

# make_growth_plot("mcpic1")
# make_growth_plot("ecpic1")
# make_growth_plot("ecpic2")
# make_growth_plot("ecpic2_new"; title="ecpic2 w/ 5 pt field solve")
# make_growth_plot("ecpic1_five"; title="ecpic1 w/ 4th order solve")
# make_growth_plot("ecpic2_five"; title="ecpic2 w/ 4th order solve")
# make_growth_plot("pics")

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
df = CSV.read("data/algo=pics_nw=1.csv", DataFrame)
# make_fit_plot(df)
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

    norm_wavenumbers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
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

    if long_data
        measured_plasma_freqs = Float64[]
        measured_errors = Float64[]
        for nw in norm_wavenumbers
            # df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)
            df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)

            exact_plasma_freq = 5.685352436149611e8
            # measured_freq = measure_plasma_freq_fft(df[!, :norm_time], df[!, :field_energy])
            # measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :field_energy])
            measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :mode_amp])

            push!(measured_plasma_freqs, measured_freq)
            push!(measured_errors, abs(measured_freq / exact_plasma_freq - 1))
        end

        scatter!(ax1, wavenumbers, measured_errors, color=:red, marker=:cross, label="Small dt")
    end

    if ppc_data
        norm_wavenumbers = [64, 128, 256]
        wavenumbers = norm_wavenumbers ./ 512 .* pi

        measured_plasma_freqs = Float64[]
        measured_errors = Float64[]
        for nw in norm_wavenumbers
            # df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)
            df = CSV.read("data/algo=$(algo)_nw=$(nw)_ppc.csv", DataFrame)

            exact_plasma_freq = 5.685352436149611e8
            # measured_freq = measure_plasma_freq_fft(df[!, :norm_time], df[!, :field_energy])
            # measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :field_energy])
            measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :mode_amp])

            push!(measured_plasma_freqs, measured_freq)
            push!(measured_errors, abs(measured_freq / exact_plasma_freq - 1))
        end

        scatter!(ax1, wavenumbers, measured_errors, color=:green, marker=:xcross, label="Large PPC")
    end

    if small_perturb_data
        norm_wavenumbers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        wavenumbers = norm_wavenumbers ./ 512 .* pi

        measured_plasma_freqs = Float64[]
        measured_errors = Float64[]
        for nw in norm_wavenumbers
            # df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)
            df = CSV.read("data/algo=$(algo)_nw=$(nw).csv_small_perturb", DataFrame)

            exact_plasma_freq = 5.685352436149611e8
            # measured_freq = measure_plasma_freq_fft(df[!, :norm_time], df[!, :field_energy])
            # measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :field_energy])
            measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :mode_amp])

            push!(measured_plasma_freqs, measured_freq)
            push!(measured_errors, abs(measured_freq / exact_plasma_freq - 1))
        end

        scatter!(ax1, wavenumbers, measured_errors, color=:orange, marker=:hline, label="Small perturb")
    end

    if small_perturb_data
        norm_wavenumbers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        wavenumbers = norm_wavenumbers ./ 512 .* pi

        measured_plasma_freqs = Float64[]
        measured_errors = Float64[]
        for nw in norm_wavenumbers
            # df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)
            df = CSV.read("data/algo=$(algo)_nw=$(nw)_very_small_perturb.csv", DataFrame)

            exact_plasma_freq = 5.685352436149611e8
            # measured_freq = measure_plasma_freq_fft(df[!, :norm_time], df[!, :field_energy])
            # measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :field_energy])
            measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :mode_amp])

            push!(measured_plasma_freqs, measured_freq)
            push!(measured_errors, abs(measured_freq / exact_plasma_freq - 1))
        end

        scatter!(ax1, wavenumbers, measured_errors, color=:yellow, marker=:vline, label="Very small perturb")
    end

    if small_perturb_data
        norm_wavenumbers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        wavenumbers = norm_wavenumbers ./ 512 .* pi

        measured_plasma_freqs = Float64[]
        measured_errors = Float64[]
        for nw in norm_wavenumbers
            # df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)
            df = CSV.read("data/algo=$(algo)_nw=$(nw)_med_perturb.csv", DataFrame)

            exact_plasma_freq = 5.685352436149611e8
            # measured_freq = measure_plasma_freq_fft(df[!, :norm_time], df[!, :field_energy])
            # measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :field_energy])
            measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :mode_amp])

            push!(measured_plasma_freqs, measured_freq)
            push!(measured_errors, abs(measured_freq / exact_plasma_freq - 1))
        end

        scatter!(ax1, wavenumbers, measured_errors, color=:black, marker=:vline, label="Medium perturb")
    end

    if small_perturb_data
        norm_wavenumbers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        wavenumbers = norm_wavenumbers ./ 512 .* pi

        measured_plasma_freqs = Float64[]
        measured_errors = Float64[]
        for nw in norm_wavenumbers
            # df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)
            df = CSV.read("data/algo=$(algo)_nw=$(nw)_alt.csv", DataFrame)

            exact_plasma_freq = 5.685352436149611e8
            # measured_freq = measure_plasma_freq_fft(df[!, :norm_time], df[!, :field_energy])
            # measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :field_energy])
            measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :mode_amp])

            push!(measured_plasma_freqs, measured_freq)
            push!(measured_errors, abs(measured_freq / exact_plasma_freq - 1))
        end

        scatter!(ax1, wavenumbers, measured_errors, color=:blue, marker=:cross, label="Alt")
    end

    if small_perturb_data
        norm_wavenumbers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        wavenumbers = norm_wavenumbers ./ 512 .* pi

        measured_plasma_freqs = Float64[]
        measured_errors = Float64[]
        for nw in norm_wavenumbers
            # df = CSV.read("data/algo=$(algo)_nw=$(nw)_long.csv", DataFrame)
            df = CSV.read("data/algo=$(algo)_nw=$(nw)_alt3.csv", DataFrame)

            exact_plasma_freq = 5.685352436149611e8
            # measured_freq = measure_plasma_freq_fft(df[!, :norm_time], df[!, :field_energy])
            # measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :field_energy])
            measured_freq = measure_plasma_freq_fit(df[!, :norm_time], df[!, :mode_amp])

            push!(measured_plasma_freqs, measured_freq)
            push!(measured_errors, abs(measured_freq / exact_plasma_freq - 1))
        end

        scatter!(ax1, wavenumbers, measured_errors, color=:orange, marker=:cross, label="Alt small dt")
    end

    # axislegend(ax1, position=:lt)

    save("$(algo)_convergence.pdf", fig)
end

# make_convergence_plot("mcpic1"; scale_line=[1, 2])
# make_convergence_plot("ecpic1")
# make_convergence_plot("ecpic2"; scale_line=[1, 2])
# make_convergence_plot("ecpic2_new"; scale_line=[1, 2, 3, 4], long_data=true, ppc_data=true, small_perturb_data=true)
# make_convergence_plot("ecpic1_five", scale_line=[1, 2])
# make_convergence_plot("ecpic2_five", scale_line=[1, 2])


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
