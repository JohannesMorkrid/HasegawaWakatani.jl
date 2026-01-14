# ------------------------------------------------------------------------------------------
#                                    Spatial Derivatives                                    
# ------------------------------------------------------------------------------------------

# ------------------------------------ Fourier Domain --------------------------------------

# -------------------------------- Construction Interface ----------------------------------

function build_operator(::Val{:diff_x}, domain::Domain; order=1, kwargs...)
    ElwiseOperator(im .* transpose(domain.kx); order=order)
end

function build_operator(::Val{:diff_xx}, domain::Domain; order=1, kwargs...)
    build_operator(Val(:diff_x), domain; order=2 * order)
end

function build_operator(::Val{:diff_y}, domain::Domain; order=1, kwargs...)
    ElwiseOperator(im .* domain.ky; order=order)
end

function build_operator(::Val{:diff_yy}, domain::Domain; order=1, kwargs...)
    build_operator(Val(:diff_y), domain; order=2 * order)
end

# Helper function 
function get_laplacian(domain::Domain)
    ks = wave_vectors(domain)
    N = length(ks)
    reshaped = ntuple(i -> reshape(ks[i], ntuple(j -> j == i ? length(ks[i]) : 1, N)), N)
    return -mapreduce(k -> k .^ 2, (a, b) -> a .+ b, reshaped)
end

function build_operator(::Val{:laplacian}, domain::Domain; order=1, kwargs...)
    ElwiseOperator(get_laplacian(domain); order=order)
end

# ----------------------------------- Chebyshev Domain -------------------------------------

# Chebyshev
# function chebyshev_differentiation_matrix(x::AbstractVector)
#     Nx = length(x)
#     D = zeros(Nx, Nx)
#     for i in eachindex(x)
#         for j in eachindex(x)
#             if i == j
#                 D[j, i] = -x[j] / (2(1 - x[j]^2))
#             else
#                 ci = i != 1 && i != Nx ? 1 : 2
#                 cj = j != 1 && j != Nx ? 1 : 2
#                 D[i, j] = (ci / cj) * (-1)^(i + j) / (x[i] - x[j])
#             end
#         end
#     end
#     D[1, 1] = (2 * (Nx - 1)^2 + 1) / 6
#     D[Nx, Nx] = -(2 * (Nx - 1)^2 + 1) / 6

#     return D
# end