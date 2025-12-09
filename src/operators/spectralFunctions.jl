# ------------------------------------------------------------------------------------------
#                                    Spectral Functions                                     
# ------------------------------------------------------------------------------------------

# ------------------------------------- Construction ---------------------------------------

struct SpectralFunction{F<:Function} <: SpectralOperator
    f::F
    out::AbstractArray # TODO specialize?
    #args, kwargs? TODO perhaps expand upon this
    quadratic_term::QuadraticTerm
    function SpectralFunction(f::Function, domain::AbstractDomain,
                              quadratic_term::QuadraticTerm)
        out = zeros(spectral_size(domain)) |> memory_type(domain)
        out = complex.(out)
        new{typeof(f)}(f, out, quadratic_term)
    end
end

function operator_dependencies(::Union{Val{:reciprocal},Val{:spectral_exp},
                                       Val{:spectral_expm1},
                                       Val{:spectral_log}}, ::Type)
    [OperatorRecipe(:quadratic_term)]
end

function build_operator(::Val{:reciprocal}, domain::AbstractDomain; quadratic_term,
                        kwargs...)
    SpectralFunction(u -> div(1, u), domain, quadratic_term)
end

function build_operator(::Val{:spectral_exp}, domain::AbstractDomain; quadratic_term,
                        kwargs...)
    SpectralFunction(exp, domain, quadratic_term)
end

function build_operator(::Val{:spectral_expm1}, domain::AbstractDomain; quadratic_term,
                        kwargs...)
    SpectralFunction(expm1, domain, quadratic_term)
end

function build_operator(::Val{:spectral_log}, domain::AbstractDomain; quadratic_term,
                        kwargs...)
    SpectralFunction(log, domain, quadratic_term)
end

# -------------------------------------- Main Method ---------------------------------------

@inline function (op::SpectralFunction)(u::AbstractArray)
    out = op.out
    #println(typeof(out), size(out))
    @unpack U, V, up, padded, transforms, dealiasing_coefficient = op.quadratic_term
    mul!(U, bwd(transforms), padded ? pad!(up, u, typeof(transforms)) : u)
    V .= op.f.(dealiasing_coefficient * U)
    mul!(padded ? up : out, fwd(transforms), V)
    #println(typeof(out), size(out))
    #println(typeof(up), size(up))
    #println(typeof(transforms))
    padded ? unpad!(out, up, typeof(transforms)) / dealiasing_coefficient : up
end