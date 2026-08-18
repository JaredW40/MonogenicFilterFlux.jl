module CUDAExt

#=  CUDA support for MonoConvFFT.

    This extension is small on purpose. All of the numerics in
    src/MonogenicTransform.jl are already device-agnostic - they are written
    against `AbstractArray`, `plan * x`, `inv(plan) * y` and broadcasts, every
    one of which cuFFT and CUDA.jl implement. What is *not* device-agnostic is
    moving a layer between devices, and that is all that lives here.

    The one genuinely hard part is that a fft plan cannot be device-copied.
    `Adapt`ing a plan's fields gives us a struct that looks like a plan and
    holds host pointers, but applying it is undefined behaviour. Plans have 
    to be rebuilt from their geometry on the target device, exactly as 
    ConvFFT was in FourierFilterFlux.

    Note: FourierFilterFlux is a hard dependency of MonogenicFilterFlux and 
    its own CUDAExt has identical triggers, so it is guaranteed to be loaded 
    whenever this file is. It already provides `Functors.@leaf` on the FFTW 
    plan types, `AbstractFFTs.AdjointStyle(::cuFFT.CuFFTPlan)`, 
    `CUDA.cu(::FFTW.cFFTWPlan)` and `Adapt.adapt(::CPUDevice, ::cuFFT.CuFFTPlan)`. 
    Redefining any of those here would be the same method signature defined 
    twice in two packages - a "method definition overwritten" warning at 
    precompile time and a load order dependency for which definition wins. 
    The plan builders below are private to this module for the same reason. =#

using MonogenicFilterFlux
using MonogenicFilterFlux: MonoConvFFT, _rebuild, _cpuPlan, convDims
using CUDA, cuFFT
using Adapt
using AbstractFFTs, FFTW
import MLDataDevices: CUDADevice, CPUDevice

#=  Rebuild a plan on the device. `size(plan)` is the backend-independent
    accessor for the padded buffer shape; the transformed region comes from the
    layer's `D` type parameter rather than the plan, since only FFTW exposes
    `.region`. `complex(...)` is idempotent here - `plan_fft` complexifies
    real input, so a MonoConvFFT plan is always a complex plan - and is kept
    as cheap insurance against that changing. =#
_gpuPlan(p::Nothing, region) = nothing
_gpuPlan(p::Tuple, region) = map(pi -> _gpuPlan(pi, region), p)
_gpuPlan(p, region) = cuFFT.plan_fft(CUDA.zeros(complex(eltype(p)), size(p)...), region)

"""
    adapt_structure(::CUDADevice, m::MonoConvFFT)

Move a monogenic layer onto the GPU: `CuArray` the filter bank, rebuild the
plan with cuFFT. Reached by `gpu(m)`, `m |> gpu`, and `cu(m)`.
"""
function Adapt.adapt_structure(dev::CUDADevice,
    m::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}) where {D,OT,F,A,PD,P,T,S,AL}
    if !CUDA.functional()
        @warn "CUDA is loaded but not functional -- leaving the MonoConvFFT on the CPU." maxlog = 1
        return m
    end
    #=  `CuArray` rather than `adapt(dev, ...)`: MLDataDevices routes CUDADevice
        through `CUDA.cu`, which silently demotes Float64 to Float32. That would
        be fine on its own, but the plan below is built from the *plan's* eltype,
        so a demoted filter bank and an undemoted plan would disagree and send
        every fft through AbstractFFTs' host-allocating `copy1` fallback. =#
    w = CuArray(m.weight)
    p = _gpuPlan(m.fftPlan, convDims(m))
    return _rebuild(m, w, p)
end

# Already on the device: rebuilding the plan would be correct but wasteful.
Adapt.adapt_structure(::CUDADevice,
    m::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}) where {D,OT,F,A<:CuArray,PD,P,T,S,AL} = m

#=  The CPU direction. The `A<:CuArray` bound is load bearing: MetalExt defines
    a method for this same `::CPUDevice` first argument, and the weight type is
    the only thing that tells the two apart. =#
function Adapt.adapt_structure(::CPUDevice,
    m::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}) where {D,OT,F,A<:CuArray,PD,P,T,S,AL}
    w = adapt(Array, m.weight)
    p = _cpuPlan(m)
    return _rebuild(m, w, p)
end

# `adapt(CuArray, m)` / `adapt(Array, m)` spellings, for symmetry with the
# array-type-based Adapt calls elsewhere in the ecosystem.
Adapt.adapt_structure(::Type{<:CuArray}, m::MonoConvFFT) = Adapt.adapt_structure(CUDADevice(), m)
Adapt.adapt_structure(::Type{<:Array}, m::MonoConvFFT{D,OT,F,A}) where {D,OT,F,A<:CuArray} =
    Adapt.adapt_structure(CPUDevice(), m)

#=  Restores `cu(layer)`, which was removed when the plan could not be moved.
    Not piracy: MonoConvFFT is our type. =#
CUDA.cu(m::MonoConvFFT) = Adapt.adapt_structure(CUDADevice(), m)

end # module
