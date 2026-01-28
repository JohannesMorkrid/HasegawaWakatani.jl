# ------------------------------------------------------------------------------------------
#                                    Solve Phi Operator                                     
# ------------------------------------------------------------------------------------------

abstract type SolvePhi <: SpectralOperator end

# Three cases:
# 1) Boussinesq approximation:
struct SolvePhiSimplified{T<:AbstractArray} <: SolvePhi
    laplacian_inv::T
    function SolvePhiSimplified(domain)
        laplacian = get_laplacian(domain)
        laplacian_inv = laplacian .^ -1
        @allowscalar laplacian_inv[1] = 0 # First entry will always be NaN or Inf
        new{typeof(laplacian_inv)}(laplacian_inv)
    end
end

# 2) Non-Boussinesq, no relaxation:
struct SolvePhiNonBoussinesq{T<:AbstractArray,S<:AbstractArray,P<:AbstractArray} <: SolvePhi
    laplacian_inv::T
    diff_x::LinearOperator
    diff_y::LinearOperator
    quadratic_term::QuadraticTerm
    C1::S
    C2::S
    phi::S
    N::P
    dηdx::P
    dηdy::P
    atol::Number
    rtol::Number
    maxiters::Number

    function SolvePhiNonBoussinesq(domain, diff_x, diff_y, quadratic_term;
                                   atol, rtol, maxiters)
        laplacian = get_laplacian(domain)
        laplacian_inv = laplacian .^ -1
        @allowscalar laplacian_inv[1] = 0 # First entry will always be NaN or Inf

        C1 = zeros(spectral_size(domain)) |> domain.MemoryType{complex(domain.precision)}
        C2 = zero(C1)
        phi = zero(C1)
        N = zero(quadratic_term.U)
        dηdx = zero(N)
        dηdy = zero(N)
        new{typeof(laplacian_inv),typeof(C1),typeof(N)}(laplacian_inv, diff_x, diff_y,
                                                        quadratic_term, phi, C1, C2, N,
                                                        dηdx, dηdy, atol, rtol, maxiters)
    end
end

# 3) Non-Boussinesq, with relaxation:
struct SolvePhiRelaxation{T<:AbstractArray,G<:AbstractArray} <: SolvePhi
    laplacian_inv::T
    gdg::G
    reciprocal::SpectralFunction
    grad_dot_grad::GradDotGradOperator

    function SolvePhiRelaxation(domain, reciprocal, grad_dot_grad; boussinesq=true)
        laplacian = get_laplacian(domain)
        laplacian_inv = laplacian .^ -1
        @allowscalar laplacian_inv[1] = 0 # First entry will always be NaN or Inf

        if boussinesq
            gdg = []
        else
            gdg = complex(similar(laplacian))
        end

        new{typeof(laplacian_inv),typeof(gdg)}(laplacian_inv, gdg, reciprocal,
                                               grad_dot_grad)
    end
end

# The simplified case will discard these dependencies
function operator_dependencies(::Val{:solve_phi}, ::Type{_}) where {_}
    [OperatorRecipe(:diff_x), OperatorRecipe(:diff_y), OperatorRecipe(:quadratic_term)]
end

# ------------------------------------- Constructors ---------------------------------------

# General user-interface:
function build_operator(::Val{:solve_phi}, domain::AbstractDomain; boussinesq=true,
                        relaxation=false, kwargs...)
    _build_operator(Val(:solve_phi), domain, Val(boussinesq), Val(relaxation); kwargs...)
end

# Construct simplified case
function _build_operator(::Val{:solve_phi}, domain::Domain, boussinesq::Val{true},
                         relaxation; kwargs...)
    SolvePhiSimplified(domain)
end

# Construct non-bousinesq, no-relaxation case:
function _build_operator(::Val{:solve_phi}, domain::Domain, boussinesq::Val{false},
                         relaxation::Val{false}; diff_x, diff_y, quadratic_term, atol, rtol,
                         maxiters, kwargs...)
    SolvePhiNonBoussinesq(domain, diff_x, diff_y, quadratic_term; atol, rtol, maxiters)
end

# Construct non-bousinesq, relaxation case:
function _build_operator(::Val{:solve_phi}, domain::Domain, boussinesq::Val{false},
                         relaxation::Val{true}; kwargs...)
    SolvePhiRelaxation(domain, reciprocal, grad_dot_grad)
end

# ------------------------------------- Main Methods ---------------------------------------

# In-place (boussinesq)
function (op::SolvePhiSimplified)(out::T, u::T) where {T<:AbstractArray}
    out .= op.laplacian_inv .* u
end

# Out-of-place (boussinesq)
(op::SolvePhiSimplified)(u::T) where {T<:AbstractArray} = op(similar(u), u)

# In-place (non-boussinesq)
function (op::SolvePhiNonBoussinesq)(out::T, n::T, ϖ::T) where {T<:AbstractArray}
    @unpack laplacian_inv, diff_x, diff_y, quadratic_term, C1, C2, phi, N, dηdx, dηdy = op
    @unpack atol, rtol, maxiters = op
    @unpack U, V, up, vp, padded, transforms, dealiasing_coefficient = quadratic_term

    #Initial guess ϕ₀ = ∇⁻²(ϖ/N)
    mul!(U, bwd(transforms), padded ? pad!(up, n, typeof(transforms)) : n)
    mul!(V, bwd(transforms), padded ? pad!(vp, ϖ, typeof(transforms)) : ϖ)
    @. V ./= U
    mul!(padded ? vp : C1, fwd(transforms), V)
    padded ? C1 .= unpad!(C1, vp, typeof(transforms)) ./ dealiasing_coefficient : C1
    phi .= laplacian_inv .* C1

    # Compute η = ln(N) (U = N, V = ϖ/N, phi = ϕ₀, C1 = (ϖ/N)_hat unpadded)
    N .= dealiasing_coefficient .* U
    V .= log.(N)
    mul!(padded ? vp : C1, fwd(transforms), V)
    padded ? C1 .= unpad!(C1, vp, typeof(transforms)) ./ dealiasing_coefficient : C1
    # C1 = ̂η at this stage, U = N, V = η

    # Compute ∇_⟂η, remains constant for all iterations
    diff_x(C2, C1)
    mul!(dηdx, bwd(transforms), padded ? pad!(up, C2, typeof(transforms)) : C2)
    dηdx .*= dealiasing_coefficient
    diff_y(C2, C1)
    mul!(dηdy, bwd(transforms), padded ? pad!(vp, C2, typeof(transforms)) : C2)
    dηdy .*= dealiasing_coefficient

    # To compare to ϖ
    C2 .= zero.(C2)
    out .= phi

    # Compute residuals
    res = norm(C2 - ϖ)
    prev_res = res
    iters = 0

    # Iterate to approximately solve for potential ϕ
    while res > rtol && iters < maxiters
        # Compute ∇_⟂ϕ_i then (∇_⟂η)⋅(∇_⟂ϕ_i) and sum up while computing ϖᵢ = ∇⋅(n∇ϕᵢ) 
        # 1) C1 = (∂ₓϕ₀)_hat
        diff_x(C1, out)
        # 2) U = (∂ₓϕ₀)_p
        mul!(U, bwd(transforms), padded ? pad!(up, C1, typeof(transforms)) : C1) # iFFT
        # 3) V = (N∂ₓϕ₀)_p
        V .= N .* U
        # 4) C2 = (N∂ₓϕ₀)_hat
        mul!(padded ? up : C2, fwd(transforms), V)                               # FFT
        padded ? unpad!(C2, up, typeof(transforms)) : C2
        # 5) C2 = (∂ₓ(N∂ₓϕ₀))_hat
        diff_x(C2, C2)
        # 6) V = (∂ₓη∂ₓϕ₀)_p
        V .= dηdx .* U
        # 7) C1 = (∂yϕ₀)_hat
        diff_y(C1, out)
        # 8) U = (∂yϕ₀)_p
        mul!(U, bwd(transforms), padded ? pad!(up, C1, typeof(transforms)) : C1) # iFFT
        # 9) V = (∂ₓη∂ₓϕ₀ + ∂yη∂yϕ₀)_p
        V .+= dηdy .* U
        # 10) out = (∂ₓη∂ₓϕ₀ + ∂yη∂yϕ₀)_hat
        mul!(padded ? vp : out, fwd(transforms), V)                               # FFT
        padded ? unpad!(out, vp, typeof(transforms)) : out
        # 11) U = (N∂yϕ₀)_p
        U .*= N
        # 12) C1 = (N∂yϕ₀)_hat
        mul!(padded ? up : C1, fwd(transforms), U)                               # FFT
        padded ? unpad!(C1, up, typeof(transforms)) : C1
        # 13) C1 = (∂y(N∂yϕ₀))_hat
        diff_y(C1, C1)
        # 14) C2 = ϖₘ = (∂ₓ(N∂ₓϕ₀) + ∂y(N∂yϕ₀))_hat
        C2 .+= C1
        # 15) out = Lϕₘ = ∇⁻²(∇_⟂η⋅∇_⟂)ϕ_m
        out .*= laplacian_inv
        # 16) ϕₘ₊₁ = ϕ₀ - Lϕₘ
        out .= phi .- out

        # Update residuals and iters count
        res = norm(C2 - ϖ) / length(ϖ)
        iters += 1
    end
    return out
end

# In-place (non-boussinesq, relaxation)
function (op::SolvePhiRelaxation)(out::T, η::T, ϖ::T) where {T<:AbstractArray}
    @unpack laplacian_inv, diff_x, diff_y, quadratic_term, reciprocal = op
    #Initial guess #ϕ_0 = laplacian_inv * (ϖ./n)
    out .= laplacian_inv .* ϖ

    for i in 1:20
        grad_dot_grad(gdg, η, out)
        out .= laplacian_inv .* gdg
    end
end

# Out-of-place (non-boussinesq, relaxation)
(op::SolvePhiRelaxation)(η::T, ϖ::T) where {T<:AbstractArray} = op(out, η, ϖ)