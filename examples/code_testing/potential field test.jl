#Potential field test
include(relpath(pwd(), @__DIR__) * "/src/HasegawaWakatini.jl")

function analytical_solution(x, y; l=1)
    r2 = x^2 + y^2
    exp(-r2 / (2 * l^2))
end

function omega(x, y; l=1)
    r2 = x^2 + y^2
    (r2 - 2 * l^2) * exp(-r2 / (2 * l^2)) / (l^4)
end

d = Domain(1024, 1)
l = 0.01

Ω = omega.(d.x', d.y, l=l)
surface(Ω)

Ω_hat = d.transform.FT * Ω
ϕ_hat = solve_phi(Ω_hat, d)
ϕ = d.transform.iFT * ϕ_hat
#Remove the value at boundary since should be zero for periodicity
ϕ_n = ϕ .- ϕ[1]

ϕ_a = analytical_solution.(d.x', d.y, l=l)
norm(ϕ_n .- ϕ_a)
norm(ϕ .- ϕ_a)

surface(d, ϕ_n, xlabel="x", ylabel="y", zlabel="ϕ")
surface(d, ϕ_a)