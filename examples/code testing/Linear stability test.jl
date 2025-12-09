## Run all (alt+enter)
using HasegawaWakatani
using CUDA

# Run alternative linear stability test
domain = Domain(256, 256; Lx=48, Ly=48, MemoryType=CuArray)
ic = initial_condition(random_crossphased, domain; value=1e-3)

# Linear operator
function Linear(du, u, operators, p, t)
    @unpack laplacian = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack ν, μ = p
    dη = ν * laplacian(η)
    dΩ = μ * laplacian(Ω)
end

# Non-linear operator
function NonLinear(du, u, operators, p, t)
    @unpack solve_phi, diff_x, diff_y = operators
    η, Ω = eachslice(u; dims=3)
    dη, dΩ = eachslice(du; dims=3)
    @unpack g, σ, κ, ν = p
    ϕ = solve_phi(Ω)
    dη .= -(κ - g) * diff_y(ϕ) - g * diff_y(η) + σ * ϕ - 2 * ν * κ * diff_x(η)
    dΩ .= -g * diff_y(η) + σ * ϕ
end

# Parameters
parameters = (ν=1e-2, μ=1e-2, g=1e-1, σ=1e-1, κ=sqrt(1e1))

# Time interval
tspan = [0.0, 50.0]

# Array of diagnostics want
diagnostics = @diagnostics [
    progress(; stride=1000),
    #cfl(; stride=100, component=:radial),
    plot_density(; stride=10000),
    get_modes(; stride=100),
    get_log_modes(; stride=100)
]

# The problem
prob = SpectralODEProblem(Linear, NonLinear, ic, domain, tspan; p=parameters, dt=1e-4,
                          diagnostics=diagnostics)

# The output
output_file_name = joinpath(@__DIR__, "output", "linear-stability.h5")
output = Output(prob; filename=output_file_name, simulation_name=:parameters)

# Solve
sol = spectral_solve(prob, MSS3(), output; resume=false)

## ----------------------------------- Mode Andalysis --------------------------------------

using LaTeXStrings
using Plots
log_modes = sol.simulation["Log modes/data"]
t = sol.simulation["Log modes/t"][:]
gamma = (log_modes[:, :, end] - log_modes[:, :, end-1]) / (diff(t)[end])
@unpack g, κ, σ, ν = parameters
w0 = sqrt(g * κ) * sqrt(1 - g / κ)
plot(gamma[:, 1] / w0; xaxis=:log, xlabel=L"k_y = k_x", ylabel=L"\gamma/\gamma_0",
     ylim=[-2, 1])
vline!([(σ / ν)^(1 / 4)])

sol.simulation