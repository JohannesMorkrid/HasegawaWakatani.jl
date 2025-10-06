## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run alternative linear stability test
domain = Domain(256, 256, 100, 100, dealiased=true)
ic = initial_condition_linear_stability(domain, 1e-6)

# Linear operator (May not be static actually)
function L(u, d, p, t)
    D_η = p["D_n"] * diffusion(u, d) #.- p["g"]*diff_y(u,d)
    D_Ω = p["D_Omega"] * diffusion(u, d)
    [D_η;;; D_Ω]
end

# Non-linear operator
function N(u, d, p, t)
    η = u[:, :, 1]
    Ω = u[:, :, 2]
    ϕ = solve_phi(Ω, d)
    dη = -(1 - p["g"]) * diff_y(ϕ, d)
    dη -= p["g"] * diff_y(η, d)
    #dη .+= p["D_n"] - p["sigma_n"]
    dη -= p["sigma_n"] * ϕ
    dη -= 2 * p["D_n"] * p["kappa"] * diff_x(η, d)
    dΩ = -p["g"] * diff_y(η, d)
    dΩ += p["sigma_Omega"] * ϕ
    return [dη;;; dΩ]
end

# Parameters
parameters = Dict(
    "D_Omega" => 1e-2,
    "D_n" => 1e-2,
    "g" => 1e-3,
    "sigma_Omega" => 1e-5,
    "sigma_n" => 1e-5,
    "kappa" => sqrt(1e-3)
)

# Time interval
t_span = [0, 500]

# The problem
prob = SpectralODEProblem(L, N, ic, domain, t_span, p=parameters, dt=1e-3)

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    #RadialCOMDiagnostic(),
    ProgressDiagnostic(1000),
    #CFLDiagnostic(),
    #RadialCFLDiagnostic(100),
    PlotDensityDiagnostic(10000),
    #GetModeDiagnostic(100),
    GetLogModeDiagnostic(100),
]

# The output
output = Output(prob, 1001, diagnostics) #progressDiagnostic

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

## Save data
using JLD
save("linearstability.jld", "data", sol.u)

## Analyze data
data = Array(reshape(reduce(hcat, sol.u), size(sol.u[1])..., length(sol.u)))
data = Matrix{Number}(data')

# Unreliable atm
#data = Array(reshape(reduce(hcat, sol.u), size(sol.u[1])..., length(sol.u)))
kappa = sqrt(parameters["sigma_n"] / parameters["D_n"])
n0 = initial_condition(exponential_background, domain, kappa=kappa)
data_hat = zeros(ComplexF64, size(sol.u[1])..., length(sol.u))
for i in eachindex(sol.u)
    data_hat[:, :, 1, i] = fft(sol.u[i][:, :, 1])#fft(n0.*sol.u[i][:, :, 1])
    data_hat[:, :, 2, i] = fft(sol.u[i][:, :, 2])
end

plot(log.(abs.(data_hat[60, 60, 1, :])))
plot(domain.x, n0[1, :] + 1e4 * sol.u[end][1, :, 1])

plot(log.(abs.(data_hat[14, 14, 1, :])))

cd(relpath(@__DIR__, pwd()))
fid = h5open("output/linear-stability test.h5", "r")
simulation = fid[keys(fid)[1]]

simulation["Log mode diagnostic/data"][:, :, :]
simulation["Log mode diagnostic/t"][:]