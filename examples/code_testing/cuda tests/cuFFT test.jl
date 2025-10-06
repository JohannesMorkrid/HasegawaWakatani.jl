using CUDA
using LinearAlgebra
using FFTW
N = 2^10
A = rand(Float64,N,N)
CuArray(A)
FT = plan_rfft(A)
B = FT*A
iFT = plan_irfft(B, N)

function test!(FT, iFT, A, B)
    #mul!(B, FT, A)
    #mul!(A, iFT, B)
end

CUFFT.cufftExecR2C(FT, A, B)

#CUDA.@profile trace=true test(FT,iFT,A,B)

@cuda threads= 256 test!(FT,iFT,A,B)

#kernel = @cuda launch=false test!(FT,iFT,A,B)
#config = launch_configuration(kernel.fun)
#threads = min(N, config.threads)
#blocks = cld(N, threads)

# --------------------------------------- Debug --------------------------------------------

CUDA.versioninfo()

"""If you're running under a profiler, this situation is expected. Otherwise,
│  │ ensure that your library path environment variable (e.g., `PATH` on Windows
│  │ or `LD_LIBRARY_PATH` on Linux) does not include CUDA library paths."""

"""CUDA runtime 11.8, artifact installation
CUDA driver 11.4
NVIDIA driver 470.256.2

CUDA libraries: 
- CUBLAS: 11.5.4
- CURAND: 10.3.0
- CUFFT: 10.9.0
- CUSOLVER: 11.4.1
- CUSPARSE: 11.7.5
- CUPTI: 2022.3.0 (API 18.0.0)
- NVML: 11.0.0+470.256.2

Julia packages: 
- CUDA: 5.6.1
- CUDA_Driver_jll: 0.10.4+0
- CUDA_Runtime_jll: 0.15.5+0

Toolchain:
- Julia: 1.11.3
- LLVM: 16.0.6

2 devices:
  0: NVIDIA GeForce GTX 1660 SUPER (sm_75, 5.781 GiB / 5.797 GiB available)
  1: Quadro P2200 (sm_61, 4.932 GiB / 4.940 GiB available)"""