module HasegawaWakatani

using FFTW, HDF5, H5Zblosc, LinearAlgebra, LaTeXStrings, MuladdMacro, UnPack, Base.Threads,
    Dates, Printf, CUDA, Adapt, Logging
export @unpack

# TODO make ext
using Plots

include("domains/domain.jl")
export Domain, diff_x, diff_xx, diff_y, diff_yy, poisson_bracket, solve_phi, quadratic_term,
    diffusion, laplacian, Δ, SpectralOperatorCache, reciprocal, spectral_exp, spectral_expm1,
    spectral_log, hyper_diffusion

include("spectralODEProblem.jl")
export SpectralODEProblem

include("schemes.jl")
export MSS1, MSS2, MSS3

include("diagnostics/diagnostics.jl")
export CFLDiagnostic, RadialCFLDiagnostic, BurgerCFLDiagnostic, RadialCOMDiagnostic,
    PlotDensityDiagnostic, PlotPotentialDiagnostic, PlotVorticityDiagnostic,
    PotentialEnergyDiagnostic, KineticEnergyDiagnostic, TotalEnergyDiagnostic,
    EnstropyEnergyDiagnostic, ResistiveDissipationDiagnostic, PotentialDissipationDiagnostic,
    KineticDissipationDiagnostic, ViscousDissipationDiagnostic, EnergyEvolutionDiagnostic,
    RadialFluxDiagnostic, ProbeDensityDiagnostic, ProbePotentialDiagnostic,
    ProbeVorticityDiagnostic, ProbeRadialVelocityDiagnostic, ProbeAllDiagnostic,
    radial_density_profile, poloidal_density_profile, radial_vorticity_profile,
    poloidal_vorticity_profile, poloidal_vorticity_profile, ProgressDiagnostic,
    GetModeDiagnostic, GetLogModeDiagnostic, RadialPotentialEnergySpectraDiagnostic,
    PoloidalPotentialEnergySpectraDiagnostic, RadialKineticEnergySpectraDiagnostic,
    PoloidalKineticEnergySpectraDiagnostic, plot_frequencies

include("outputer.jl")
export Output

include("spectralSolve.jl")
export spectral_solve

include("utilities.jl")
export initial_condition, gaussian, log_gaussian, sinusoidal, sinusoidalX, sinusoidalY,
    gaussianWallX, gaussianWallY, exponential_background, randomIC, random_phase,
    random_crossphased, isolated_blob, remove_zonal_modes, remove_streamer_modes,
    remove_asymmetric_modes!, remove_nothing, send_mail

end