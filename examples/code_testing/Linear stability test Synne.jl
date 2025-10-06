## Run all (alt+enter)
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

## Run linear stability test
domain = Domain(256, 256, 100, 100, dealiased=true)
ic = initial_condition_linear_stability(domain, 1e-6)

# Linear operator (May not be static actually)
function L(u, d, p, t)
    D_η = p["D_n"] * diffusion(u, d) #.- p["g"]*diff_y(u,d) .- p["sigma_n"]
    D_Ω = p["D_Omega"] * diffusion(u, d) #.+ p["sigma_Omega"]*solve_phi(u,d)
    [D_η;;; D_Ω]
end

# Non-linear operator
function N(u, d, p, t)
    η = @view u[:, :, 1]
    Ω = @view u[:, :, 2]
    ϕ = solve_phi(Ω, d)
    dη = -(p["kappa"] - p["g"]) * diff_y(ϕ, d)
    dη .-= p["g"] * diff_y(η, d)
    dη .-= p["sigma_n"] * η
    #dη .+= p["sigma_n"] * ϕ # This is an additional term that Synne paper neglected
    dΩ = -p["g"] * diff_y(η, d)
    dΩ .+= p["sigma_Omega"] * ϕ
    return [dη;;; dΩ]
end

# Parameters
parameters = Dict(
    "D_Omega" => 1e-2,
    "D_n" => 1e-2,
    "g" => 1e-3,
    "sigma_Omega" => 1e-5,
    "sigma_n" => 1e-5,
    "kappa" => sqrt(1e-3),
)

# Time interval
t_span = [0, 3600]

# The problem
prob = SpectralODEProblem(L, N, ic, domain, t_span, p=parameters, dt=1e-2)

# Array of diagnostics want
diagnostics = [
    #ProbeDensityDiagnostic([(5, 0), (8.5, 0), (11.25, 0), (14.375, 0)], N=10),
    ProgressDiagnostic(100),
    CFLDiagnostic(),
    PlotDensityDiagnostic(1000),
    GetModeDiagnostic(100),
    GetLogModeDiagnostic(100, 1), # Corresponds to kx = 0
]

# The output
cd(relpath(@__DIR__, pwd()))
output = Output(prob, 1001, diagnostics, "output/linear-stability test Synne.h5", simulation_name=:parameters)

FFTW.set_num_threads(16)

## Solve and plot
sol = spectral_solve(prob, MSS3(), output)

# ------------------ Mode analysis ---------------------------------------------------------

log_modes = stack(sol.diagnostics[end].data)
gamma = (log_modes[:, :, end] - log_modes[:, :, end-1]) / (100 * prob.dt)
p = parameters
w0 = sqrt(p["g"] * p["kappa"]) * sqrt(1 - p["g"] / p["kappa"])
plot(gamma[:, 1] / w0, xaxis=:log, xlabel=L"k_y = k_x", ylabel=L"\gamma/\gamma_0", ylim=[-2, 1])
vline!([p["kappa"]^(-1 / 4)])

send_mail("Linear stability (Synne) test finnished!")