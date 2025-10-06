using HasegawaWakatani
using Plots

# Let's solve a simple heat equation 
# u_t = κ Δu + ν Δu
# on a 2D periodic domain with initial condition a Gaussian bump

domain = Domain(32, 32, 10, 10, use_cuda=false)
u0 = initial_condition(gaussian, domain)

# Linear operator
function L(du, u, d, p, t)
    du .= diffusion(u, d) .+ diffusion(u, d)
end

# Non-linear operator (none)                
function N(du, u, d, p, t)
    du .= 0
end

# Time interval
tspan = [0.0, 1.0]

# The problem
prob = SpectralODEProblem(L, N, u0, domain, tspan, p=Dict("nu" => 0), dt=1e-2)

output = Output(prob, filename="heat_equation.h5", store_hdf=true)

sol = spectral_solve(prob, MSS1(), output)

anim = Animation()
for t in 1:size(sol.t, 1)
    heatmap(sol.u[t], title="Time: $t", xlabel="X", ylabel="Y", c=:viridis)
    frame(anim)
end
gif(anim, "lol.gif", fps=10)