using MonogenicFilterFlux, ContinuousWavelets, FourierFilterFlux
using Flux, FFTW, AbstractFFTs, Wavelets, Zygote
using Logging, Test, LinearAlgebra, Random

Random.seed!(1234)

#=  
        GROUP=All    (default) attempt every backend this environment has
        GROUP=CUDA   only attempt CUDA -- Metal.jl is never touched
        GROUP=Metal  only attempt Metal -- CUDA.jl is never touched
        GROUP=CPU    skip both GPU backends entirely =#
const GROUP = get(ENV, "GROUP", "All")

function _loadBackend(ex)
    Core.eval(@__MODULE__, ex)
    isdefined(Base, :retry_load_extensions) && Base.retry_load_extensions()
    return true
end

const HAS_CUDA = if GROUP in ("All", "CUDA")
    try
        _loadBackend(:(using CUDA, cuDNN, cuFFT))
    catch e
        @info "CUDA/cuDNN/cuFFT not available here -- skipping CUDATests.jl" exception = e
        false
    end
else
    false
end

const HAS_METAL = if GROUP in ("All", "Metal")
    try
        _loadBackend(:(using Metal))
    catch e
        @info "Metal not available here -- skipping MetalTests.jl" exception = e
        false
    end
else
    false
end

@info "backend loading" GROUP HAS_CUDA HAS_METAL has_gpu = MonogenicFilterFlux.has_gpu() cuda_ext = Base.get_extension(MonogenicFilterFlux, :CUDAExt) !== nothing metal_ext = Base.get_extension(MonogenicFilterFlux, :MetalExt) !== nothing

@testset "MonogenicFilterFlux.jl" begin
    include("MonogenicConvFFTConstructors.jl")
    include("MonogenicTransformTests.jl")
    include("deviceMovementTests.jl")
    include("extensionStubTests.jl")

    HAS_CUDA && include("CUDATests.jl")
    HAS_METAL && include("MetalTests.jl")
end