# Fundamental building block for Advectra code
struct Field{T,N,D<:AbstractDomain,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    domain::D
end

Base.size(field::Field) = size(field.data)
Base.parent(field::Field) = field.data
Base.IndexStyle(::Type{<:Field{T,N,D,A}}) where {T,N,D,A} = IndexStyle(A)

# TODO what is faster?
Base.@propagate_inbounds Base.getindex(field::Field, i::Int) = getindex(field.data, i)
#Base.@propagate_inbounds Base.getindex(field::Field, I::...) = getindex(field.data, I...)
Base.@propagate_inbounds Base.setindex!(field::Field, v, i::Int) = setindex!(field.data, v, i)
#Base.@propagate_inbounds Base.setindex!(field::Field, v, I...) = setindex!(field.data, v, I...)

function Base.similar(field::Field, ::Type{T}, dims::Dims) where T
    Field(similar(field.data, T, dims), field.domain)
end

function Base.showarg(io::IO, field::Field, toplevel)
    print(io, "Field(", typeof(field.data), ", ", typeof(field.domain), ")")
end

# To be compatible with GPU Arrays
Base.print_array(io::IO, F::Field) = Base.print_array(io, F.data)
Base.dataids(F::Field) = Base.dataids(F.data)

# -------------------------------- Broadcasting Machinery ----------------------------------

import Base.Broadcast: BroadcastStyle, Broadcasted

# Construction related
struct FieldStyle{S<:BroadcastStyle} <: BroadcastStyle end
FieldStyle(s::BroadcastStyle) = FieldStyle{typeof(s)}()
BroadcastStyle(::Type{<:Field{T,N,D,A}}) where {T,N,D,A} = FieldStyle(BroadcastStyle(A))

# Combination rules: two Fields combine via their inner styles
BroadcastStyle(a::FieldStyle{S1}, ::FieldStyle{S2}) where {S1,S2} =
    FieldStyle(Broadcast.result_style(S1(), S2()))
BroadcastStyle(::FieldStyle{S}, other::BroadcastStyle) where {S} =
    FieldStyle(Broadcast.result_style(S(), other))

# Taken from manual: https://docs.julialang.org/en/v1/manual/interfaces/#Broadcast-Styles
find_field(bc::Broadcasted) = find_field(bc.args)
find_field(args::Tuple) = find_field(find_field(args[1]), Base.tail(args))
find_field(x) = x
find_field(::Tuple{}) = nothing
find_field(f::Field, rest) = f
find_field(::Any, rest) = find_field(rest)

# Strip Field wrappers out of the expression tree to get inner style
@inline _unwrap(x) = x # Generic catch all
@inline _unwrap(x::Field) = parent(x)
@inline _unwrap(bc::Broadcasted{FieldStyle{S}}) where {S} =
    Broadcasted{S}(bc.f, map(_unwrap, bc.args), bc.axes)

# Allocation of output Field
function Base.similar(bc::Broadcasted{FieldStyle{S}}, ::Type{ElType}) where {S,ElType}
    field = find_field(bc) # Get field to preserve Domain
    Field(similar(_unwrap(bc), ElType), field.domain)
end

# Usefull for GPU kernel-fusion
@inline function Base.copyto!(dest::Field, bc::Broadcasted{FieldStyle{S}}) where {S}
    copyto!(dest.data, _unwrap(bc))
    dest
end

# ------------------------------- LinearAlgebra Forwarding ---------------------------------

_rewrap(result::AbstractArray, template::Field) = Field(result, template.domain)
_rewrap(result, template) = result # scalars, etc. pass through untouched

#using LinearAlgebra
for f in (:mul!, :ldiv!, :rdiv!, :dot, :norm, :tr, :det, :inv, :cross, :qr, :lu, :cholesky)
    @eval function LinearAlgebra.$f(A::Field, args...; kwargs...)
        result = LinearAlgebra.$f(parent(A), map(_unwrap, args)...; kwargs...)
        _rewrap(result, A)
    end
end

# ---------------------------------- Adapt Compatibility -----------------------------------

Adapt.adapt_structure(to, field::Field) = Field(Adapt.adapt(to, field.data), field.domain)
