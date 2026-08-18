module MetalExt

#=  Metal (Apple Silicon) support for MonoConvFFT =#

using MonogenicFilterFlux
using MonogenicFilterFlux: MonoConvFFT, _rebuild, _cpuPlan, convDims
using Metal
using Adapt
using AbstractFFTs, FFTW
import MLDataDevices: MetalDevice, CPUDevice

AbstractFFTs.AdjointStyle(::Metal.MtlFFTPlan) = AbstractFFTs.FFTAdjointStyle()
 
_gpuPlan(p::Nothing, region) = nothing
_gpuPlan(p::Tuple, region) = map(pi -> _gpuPlan(pi, region), p)
_gpuPlan(p, region) = plan_fft(Metal.zeros(complex(eltype(p)), size(p)...), region)
 
function _checkMetalPrecision(m::MonoConvFFT)
    real(eltype(m.weight)) === Float64 && error(
        "MonoConvFFT: this layer's filter bank is Float64 and Metal has no " *
        "Float64 support -- rebuild the layer with `dType = Float32` " *
        "(the MonogenicLayer default) before moving it onto a Metal device.")
    return nothing
end

"""
    adapt_structure(::MetalDevice, m::MonoConvFFT)

Move a monogenic layer onto a Metal device: `MtlArray` the filter bank, rebuild
the plan with Metal's fft. Reached by `gpu(m)`, `m |> gpu`, and `mtl(m)`.
"""
function Adapt.adapt_structure(dev::MetalDevice,
    m::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}) where {D,OT,F,A,PD,P,T,S,AL}
    if !Metal.functional()
        @warn "Metal is loaded but not functional -- leaving the MonoConvFFT on the CPU." maxlog = 1
        return m
    end
    _checkMetalPrecision(m)
    w = MtlArray(m.weight)
    p = _gpuPlan(m.fftPlan, convDims(m))
    return _rebuild(m, w, p)
end

# Already on the device.
Adapt.adapt_structure(::MetalDevice,
    m::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}) where {D,OT,F,A<:MtlArray,PD,P,T,S,AL} = m
 
# `A<:MtlArray` is what lets this coexist with CUDAExt's `::CPUDevice` method.
function Adapt.adapt_structure(::CPUDevice,
    m::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}) where {D,OT,F,A<:MtlArray,PD,P,T,S,AL}
    w = adapt(Array, m.weight)
    p = _cpuPlan(m)
    return _rebuild(m, w, p)
end

Adapt.adapt_structure(::Type{<:MtlArray}, m::MonoConvFFT) = Adapt.adapt_structure(MetalDevice(), m)
Adapt.adapt_structure(::Type{<:Array}, m::MonoConvFFT{D,OT,F,A}) where {D,OT,F,A<:MtlArray} =
    Adapt.adapt_structure(CPUDevice(), m)
 
Metal.mtl(m::MonoConvFFT) = Adapt.adapt_structure(MetalDevice(), m)
 
end # module