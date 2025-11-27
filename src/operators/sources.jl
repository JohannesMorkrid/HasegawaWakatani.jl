# ------------------------------------------------------------------------------------------
#                                          Sources                                          
# ------------------------------------------------------------------------------------------

struct Source <: SpectralOperator
    source::AbstractArray
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

# -------------------------------------- Main Method ---------------------------------------

@inline (op::SpectralConstant)(constant::Number) = constant .* op.delta
