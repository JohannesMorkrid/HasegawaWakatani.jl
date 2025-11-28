# ------------------------------------------------------------------------------------------
#                                          Sources                                          
# ------------------------------------------------------------------------------------------

struct ConstantSource <: SpectralOperator
    source::AbstractArray
end

function build_operator(::Val{:constant_source}, domain::AbstractDomain; shape, kwargs...)
    ConstantSource(shape(domain, kwargs...))
end

struct SpectralConstant <: SpectralOperator
    delta::AbstractArray
    function SpectralConstant(domain::Domain)
        MT = memory_type(domain)
        delta = MT(zeros(spectral_size(domain)))
        @allowscalar delta[1] = 1
        new(delta)
    end
end

function build_operator(::Val{:spectral_constant}, domain::AbstractDomain; kwargs...)
    SpectralConstant(domain)
end

struct Source <: SpectralOperator
    shape::AbstractArray
end

# ------------------------------------- Main Methods ---------------------------------------

# To allow for adding a constant
@inline (op::SpectralConstant)(constant::Number) = constant .* op.delta

# To allow for adding constant source
@inline Base.:+(field::AbstractArray, op::ConstantSource) = field .+ op.source
@inline Base.:+(op::ConstantSource, field::AbstractArray) = field .+ op.source
@inline (op::ConstantSource)() = op.source

# To allow for field dependent, i.e. density dependent sources
@inline (op::Source)(field::AbstractArray) = op.quadratic_term(op.shape, field)
@inline (op::ConstantSource)(field::AbstractArray) = op.constant .* field

# TODO allow for time and field dependent sources
#@inline (op::Source)(field::AbstractArray, time::Number) = op.quadratic_term(op.shape(domain,
#                                                                                      t),
#                                                                             field)
#@inline (op::Source)(time::Number) = (op.shape(domain, t), field)

# """
# # Field in-dependent source:
# eq... + source
# eq... + source()

# # Field dependent source:
# eq... + source(f)
# """

# function NonLinear(du, u, operators, p, t)
#     @unpack solve_phi, poisson_bracket, diff_y, quadratic_term = operators
#     @unpack spectral_exp, spectral_expm1, spectral_log, = operators
#     @unpack g, σ_0 = p
#     n, Ω = eachslice(u; dims=3)
#     dn, dΩ = eachslice(du; dims=3)
#     ϕ = solve_phi(Ω)

#     dn .= poisson_bracket(n, ϕ) - g * diff_y(n) + g * quadratic_term(n, diff_y(ϕ)) -
#           σ_0 * quadratic_term(n, spectral_exp(-ϕ)) + source(n, t)
#     dΩ .= poisson_bracket(Ω, ϕ) - g * diff_y(spectral_log(n)) - σ_0 * spectral_expm1(-ϕ)
# end

# S = ConstantSource(exp.(rand(256, 256)))
# field = rand(256, 256)

# field + S
# field + S()
# field + field