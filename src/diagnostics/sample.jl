# ------------------------------------------------------------------------------------------
#                                    Sample Diagnostics                                     
# ------------------------------------------------------------------------------------------

# ---------------------------------------- Density -----------------------------------------

sample_density(state, prob, time; kwargs...) = selectdim(state, ndims(prob.domain) + 1, 1)

function build_diagnostic(::Val{:sample_density}; kwargs...)
    Diagnostic(; name="Density",
               method=sample_density,
               metadata="Sampled density field")
end

# --------------------------------------- Vorticity ----------------------------------------

sample_vorticity(state, prob, time; kwargs...) = selectdim(state, ndims(prob.domain) + 1, 2)

function build_diagnostic(::Val{:sample_vorticity}; kwargs...)
    Diagnostic(; name="Vorticity",
               method=sample_vorticity,
               metadata="Sampled vorticity field")
end

# --------------------------------------- Potential ----------------------------------------

function sample_potential(state, prob, time; kwargs...)
    @unpack operators, domain = prob
    @unpack solve_phi = operators
    Ω = selectdim(state, ndims(domain) + 1, 2)
    ϕ = bwd(domain) * solve_phi(Ω)
    return ϕ
end

requires_operator(::Val{:sample_potential}; kwargs...) = [OperatorRecipe(:solve_phi)]

function build_diagnostic(::Val{:sample_potential}; kwargs...)
    Diagnostic(; name="Potential",
               method=sample_potential,
               metadata="Sampled potential field",
               assumes_spectral_state=true)
end