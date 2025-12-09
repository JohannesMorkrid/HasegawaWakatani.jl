# ------------------------------------------------------------------------------------------
#                                    Spectral Operators                                     
# ------------------------------------------------------------------------------------------

# ---------------------------------------- General -----------------------------------------

# Abstract class that may inherit from a more abstract one
abstract type SpectralOperator end

# ----------------------------------- Linear Operators -------------------------------------

# Abstract type that all Linear operators inherit from
abstract type LinearOperator{T} <: SpectralOperator end

# --------------------------------- Elementwise Operator -----------------------------------

struct ElwiseOperator{T<:AbstractArray} <: LinearOperator{T}
    coeffs::T
    order::Number

    ElwiseOperator(coeffs; order=1) = new{typeof(coeffs)}(coeffs .^ order, order)
end

# Out-of-place operator
@views @inline (op::ElwiseOperator)(u::AbstractArray) = op.coeffs .* u

# To be able to use @. without applying LinearOperator to array element
import Base.Broadcast: broadcasted
broadcasted(op::ElwiseOperator, x) = broadcasted(*, op.coeffs, x)

# ------------------------------------ Matrix Operator -------------------------------------

struct MatrixOperator{T<:AbstractArray} <: LinearOperator{T}
    coeffs::T
    order::Number
end

# Out-of-place operator # TODO figure out what to do here
@views @inline (op::MatrixOperator)(u::AbstractArray) = op.coeffs * u

include("spatialDerivatives.jl")

# --------------------------------- Non-Linear Operators -----------------------------------

# Abstract type that all NonLinear operators inherit from
abstract type NonLinearOperator <: SpectralOperator end

include("quadraticTerm.jl")
include("poissonBracket.jl")

# ---------------------------------------- Others ------------------------------------------

include("solvePhi.jl")
include("spectralFunctions.jl")
include("sources.jl")

# ------------------------------------ Operator Recipe -------------------------------------

struct OperatorRecipe{KwargsType<:NamedTuple}
    op::Symbol
    alias::Symbol
    kwargs::KwargsType

    function OperatorRecipe(op; alias=op, kwargs...)
        # TODO check that op is valid
        nt = NamedTuple(kwargs)
        new{typeof(nt)}(op, alias, nt)
    end
end

# TODO Figure out args
default_kwargs(::Val{T}) where {T} = NamedTuple()

Base.hash(oprecipe::OperatorRecipe, h::UInt) = hash((oprecipe.op, oprecipe.kwargs), h)
function Base.:(==)(or1::OperatorRecipe, or2::OperatorRecipe)
    or1.op == or2.op && or1.kwargs == or2.kwargs
end

# ------------------------------------- Constructors ---------------------------------------

# Catch all method, can be overwritten with specilization
operator_dependencies(::Val{_}, ::Type{__}) where {_,__} = ()

function get_operator_recipes(operators::Symbol)
    if operators == :default
        return [OperatorRecipe(:diff_x; order=1),
                OperatorRecipe(:diff_y; order=1),
                OperatorRecipe(:laplacian; order=1),
                OperatorRecipe(:solve_phi),
                OperatorRecipe(:poisson_bracket)]
    elseif operators == :SOL
        return [OperatorRecipe(:diff_x),
                OperatorRecipe(:diff_y),
                OperatorRecipe(:laplacian),
                OperatorRecipe(:solve_phi),
                OperatorRecipe(:poisson_bracket),
                OperatorRecipe(:quadratic_term)]
    elseif operators == :all
        return [OperatorRecipe(:diff_x; order=1),
                OperatorRecipe(:diff_y; order=1),
                OperatorRecipe(:laplacian; order=1),
                OperatorRecipe(:solve_phi),
                OperatorRecipe(:poisson_bracket),
                OperatorRecipe(:quadratic_term),
                OperatorRecipe(:spectral_log),
                OperatorRecipe(:spectral_exp),
                OperatorRecipe(:spectral_expm1),
                OperatorRecipe(:reciprocal)]
    elseif operators == :none
        OperatorRecipe[]
    else
        error()
    end
end

# function get_operator_recipes(operators::Symbol)
#     if operators == :default
#         return @op [diff_x(; order=1), diff_y(; order=1), laplacian(; order=1),
#                     solve_phi, poisson_bracket]
#     elseif operators == :SOL
#         return @op [diff_x, diff_y, laplacian, solve_phi, poisson_bracket, quadratic_term]
#     elseif operators == :all
#         return @op [diff_x(; order=1), diff_y(; order=1), laplacian(; order=1),
#                     solve_phi, poisson_bracket, quadratic_term, spectral_log, spectral_exp,
#                     spectral_expm1, reciprocal]
#     elseif operators == :none
#         OperatorRecipe[]
#     else
#         error()
#     end
# end

# --------------------------------- OperatorRecipe Macro -----------------------------------

# TODO implement
macro op()
end