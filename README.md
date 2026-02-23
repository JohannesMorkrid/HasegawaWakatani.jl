# HasegawaWakatani (name to be changed)

[![Build Status](https://github.com/JohannesMorkrid/HasegawaWakatani/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JohannesMorkrid/HasegawaWakatani/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JohannesMorkrid.github.io/HasegawaWakatani.jl/dev)

Code used for my master thesis to simulate sheath-interchange instability and resistive drift-wave turbulence in magnetized plasmas.
The resisitve drift-wave turbulence is described by the Hasegawa-Wakatani model

$$ \frac{\partial n}{\partial t} + \\{\phi, n\\} + \kappa\frac{\partial\phi}{\partial y} = \alpha(\phi-n) + D_n\nabla^2_\perp n $$

$$ \frac{\partial\Omega}{\partial t} + \\{\phi,\Omega\\} = \alpha(\phi-n) + D_\Omega\nabla^2_\perp\Omega $$

where $D_n$ and $D_\Omega$ may include higher order damping operators,while sheat-interchange instabilities are described by the following equations

$$\frac{\partial n}{\partial t} + \\{\phi, n\\} - gn\frac{\partial\phi}{\partial y} + g\frac{\partial n}{\partial y} = D_n\nabla^2_\perp n - \sigma_nn\exp(\Lambda-\phi) + S_n$$

$$\frac{\partial\Omega}{\partial t} + \\{\phi,\Omega\\} + g\frac{\partial\ln(n)}{\partial y} = D_\Omega\nabla^2_\perp\Omega + \sigma_\Omega[1-\exp(\Lambda-\phi)]$$

with $\Omega = \nabla^2\phi$. The code features:
* Biperiodic domain (perpendicular to $\textbf{B}$)
* Fast Fourier transform for spatial derivatives (FFTW)
* Third order stiffly stable time integrator
* HDF data output for binary format storage with blosc compression
* 2/3 Antialiasing on quadratic terms and non-linear functions
* Diagnostic modules
* CUDA support

![Alt Text](assets/blob.gif)

The code atempts to be modular and generalizable to be able to solve other spectral problems. 
Plasma Biperiodic Ode-solver for Long-time Turbulence (PlasmaBOLT)

Things want to add in future versions:
* Operators, remediscent of SciMLOperators
* In-place all the way
* Rosenbrock-Euler method for first step



