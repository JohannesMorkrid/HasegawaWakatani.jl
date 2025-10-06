## Run all (alt+enter)
# include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatani.jl")
using HasegawaWakatani

## Run scheme test
domain = Domain(256, 256, 50, 50, dealiased=true, use_cuda=false)
u0 = initial_condition(log_gaussian, domain)

# Diffusion 
function L(u, d, p, t)
    p["nu"] * diffusion(u, d)
end

function N(u, d, p, t)
    p["nu"] * (quadratic_term(diff_x(u, d), diff_x(u, d), d) .+ quadratic_term(diff_y(u, d), diff_y(u, d), d))
end

# Parameters
parameters = Dict(
    "nu" => 0.5
)

t_span = [0, 2]

prob = SpectralODEProblem(L, N, u0, domain, t_span, p=parameters, dt=0.001)

function inverse_transform!(U::V) where {V<:AbstractArray}
    @views U[:, :, 1] .= exp.(U[:, :, 1]) .- 1
end

cd(relpath(@__DIR__, pwd()))
output = Output(prob, 21, [ProgressDiagnostic(10)], "output/non-linear diffusion.h5", physical_transform=inverse_transform!)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

## Time convergence test
timesteps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
_, convergence1 = test_timestep_convergence(prob, HeatEquationAnalyticalSolution2, timesteps,
    MSS1(), physical_transform=inverse_transform!)
_, convergence2 = test_timestep_convergence(prob, HeatEquationAnalyticalSolution2, timesteps,
    MSS2(), physical_transform=inverse_transform!)
_, convergence3 = test_timestep_convergence(prob, HeatEquationAnalyticalSolution2, timesteps,
    MSS3(), physical_transform=inverse_transform!)
plot(timesteps, convergence1, xaxis=:log, yaxis=:log, label="MSS1")
plot!(timesteps, convergence2, xaxis=:log, yaxis=:log, label="MSS2", color="dark green")
plot!(timesteps, convergence3, xaxis=:log, yaxis=:log, label="MSS3", color="orange")
plot!(timesteps, 0.5 * timesteps .^ 3, linestyle=:dash, label=L"\frac{1}{2}dt^3")
plot!(timesteps, 0.5 * timesteps .^ 2, linestyle=:dash, label=L"\frac{1}{2}dt^2", xlabel="dt",
    ylabel=L"||U-u_a||", title="Timestep convergence, Nonlin-Diffusion (N =$(domain.Nx))", xticks=timesteps)
savefig("figures/Timestep convergence, Nonlin-Diffusion (N =$(domain.Nx)).pdf")

using JLD
jldopen("output/non-linear diffusion test.jld", "w") do file
    g = create_group(file, "data")
    g["convergence1"] = convergence1
    g["convergence2"] = convergence2
    g["convergence3"] = convergence3
    g["timesteps"] = timesteps
    #g["colors"] = "#".*hex.(getindex.(p.series_list[1:end], :seriescolor))
end

## Resolution convergence test
resolutions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
_, convergence1 = test_resolution_convergence(prob, log_gaussian, HeatEquationAnalyticalSolution2,
    resolutions, MSS1(), physical_transform=inverse_transform!)
_, convergence2 = test_resolution_convergence(prob, log_gaussian, HeatEquationAnalyticalSolution2,
    resolutions, MSS2(), physical_transform=inverse_transform!)
_, convergence3 = test_resolution_convergence(prob, log_gaussian, HeatEquationAnalyticalSolution2,
    resolutions, MSS3(), physical_transform=inverse_transform!)

plot(resolutions, convergence1, xaxis=:log2, yaxis=:log, label="MSS1")
plot!(resolutions, convergence2, xaxis=:log2, yaxis=:log, label="MSS2", color="dark green")
plot!(resolutions, convergence3, xaxis=:log2, yaxis=:log, label="MSS3", color="orange")
plot!(resolutions[1:end-4], 0.5 * exp.(-0.5 * resolutions)[1:end-4], label=L"\frac{1}{2}\exp\left(-\frac{N}{2}\right)", linestyle=:dash,
    xaxis=:log2, yaxis=:log, xticks=resolutions, xlabel=L"N_x \wedge N_y",
    ylabel=L"||U-u_a||/N_xN_y", title="Resolution convergence, Nonlin-Diffusion (dt=$(prob.dt))")
savefig("figures/Resolution convergence, Nonlin-Diffusion (dt=$(prob.dt)).pdf")

jldopen("output/non-linear diffusion resolution test.jld", "w") do file
    g = create_group(file, "data")
    g["convergence1"] = convergence1
    g["convergence2"] = convergence2
    g["convergence3"] = convergence3
    g["resolutions"] = resolutions
    #g["colors"] = "#".*hex.(getindex.(p.series_list[1:end], :seriescolor))
end

send_mail("Non-linear diffusion results are in!")