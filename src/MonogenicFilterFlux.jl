module MonogenicFilterFlux
using Reexport
using Zygote, Flux, LinearAlgebra
using AbstractFFTs, FFTW # TODO: check the license on FFTW and such
using ContinuousWavelets
using Adapt
using Functors
using MLDataDevices
using RecipesBase
using Base: tail
using FourierFilterFlux

#=  CUDA/cuFFT/cuDNN and Metal are *weak* dependencies now; see ext/CUDAExt.jl
    and ext/MetalExt.jl. Nothing in src/ may reference them directly. All the
    device-specific work is confined to two things the extensions provide:

      1. Adapt.adapt_structure(dev, ::MonoConvFFT) - moves weights and
         *rebuilds* the fft plan (a plan is a handle to backend state, it
         cannot be memcpy'd between devices).
      2. a backend plan constructor, used by (1).

    Note that FourierFilterFlux's own CUDAExt/MetalExt are guaranteed to be
    loaded alongside ours (as FourierFilterFlux is a hard dependency here, and
    the two extensions have the same triggers), so the `Functors.@leaf` marks
    on FFTW.rFFTWPlan/cFFTWPlan/Plan and the `Adapt.adapt(::CUDADevice,
    ::FFTW.cFFTWPlan)` style methods already exist. We deliberately do not
    repeat them here, as identical method signatures defined in two packages
    would produce "method definition overwritten" warnings at precompile time. =#

import Adapt: adapt
import MLDataDevices: CPUDevice
#=  FourierFilterFlux exports both of these, so `using FourierFilterFlux` puts
    them in scope; adding MonoConvFFT methods to them requires an explicit
    import, otherwise Julia refuses with "function ... must be explicitly
    imported to be extended". =#
import FourierFilterFlux: outType, originalDomain

export pad, poolSize, originalDomain, params!, formatJLD
export Periodic, Pad, ConvBoundary, Sym, analytic, outType, nFrames
# layer types and constructors
export ConvFFT, waveletLayer, averagingLayer
# inits
export positive_glorot_uniform, iden_perturbed_gaussian,
    uniform_perturbed_gaussian
# Analytic types
export TransformTypes, AnalyticWavelet, RealWaveletRealSignal,
    RealWaveletComplexSignal, NonAnalyticMatching
# utils
export effectiveSize, originalSize
# device movement (re-exported from Flux so `using MonogenicFilterFlux` is enough)
export gpu, cpu

include("MonogenicBoundaries.jl");

# MonoConvFFT

# (remark) Similar to ConvFFT
# (remark) but omit bias V, Boundary condition PD , Analyticity An

# (1) D: dimension
#        e.g. N - 1
# (2) OT :: DataType = Float32
#         the output datatype, usually determined by the (space domain) type of the filters.
# (3) F: (not default) Non-linearity
#         e.g. (σ :: F = typeof(abs))
# (4) A: (not default) weight
#         e.g. (weight :: typeof(w))
# (4) PD: (not default) boundary
#         e.g. (bc :: typeof(boundary))
# (5) P: (not default) Plan
#         Bool=true: use precomputed fft plan(s), as this is a significant cost.
#         Set this to "false" if you have variable batch/channel sizes.
#         e.g. (fftPlan :: typeof(fftPlan))
# (6) T: trainable
#         Bool=true: The entries are trainable as Flux objects, and so are
#         returned in a "params" call if this is "true".
# (7) S: scale
# (8) AL: averagingLayer

#e.g. MonoConvFFT{N - 1,OT,typeof(σ),typeof(w),typeof(fftPlan),trainable, typeof(scale), typeof(averagingLayer)}(σ, w, boundary, fftPlan, scale, averagingLayer)

struct MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}
    σ::F
    weight::A
    bc::PD
    fftPlan::P
    scale::S
    averagingLayer::AL
end

#=  MonoConvFFT holds an fft plan (P) and two non-array dispatch tags (bc, and
    the σ/scale/averagingLayer settings). Functors must not walk into it: a
    plan's fields are backend handles, and rebuilding the struct field-by-field
    would silently lose the `OT` and `trainable` type parameters, since neither
    is recoverable from the field values alone. Marking it a leaf routes every
    `gpu`/`cpu`/`fmap` through `Adapt.adapt_structure`, which the extensions
    implement and which preserves all the type parameters exactly. =#
Functors.@leaf MonoConvFFT

import Base: ndims
ndims(c::MonoConvFFT{D}) where {D} = D

"""
    outType(::MonoConvFFT)

The (space domain) element type the layer was built to produce.
"""
outType(::MonoConvFFT{D,OT}) where {D,OT} = OT

"""
    isTrainable(::MonoConvFFT)

Whether the filter bank is exposed to `Flux.trainable`.
"""
isTrainable(::MonoConvFFT{D,OT,F,A,PD,P,T}) where {D,OT,F,A,PD,P,T} = T

"""
    convDims(m::MonoConvFFT)

The dimensions the fft plan transforms over, i.e. `(1, ..., D)`. Needed
whenever a plan has to be rebuilt on another device, since the region is not
recoverable in a backend-independent way from the plan object itself.
"""
convDims(::MonoConvFFT{D}) where {D} = ntuple(identity, D)

#=  The "no frills" constructor. Two jobs:
      - it is the bare positional constructor `Functors`/`Flux.@functor` style
        reconstruction looks for (fields in declaration order); without a
        matching method, reconstruction fails with a MethodError rather than
        anything informative.
      - it lets the extensions and `_rebuild` build a layer without going back
        through filter construction.
    `OT` cannot be recovered from the plan (a real-input `plan_fft` and a
    complex-input `plan_fft` are both complex-eltype plans), so it is a keyword
    defaulting to the real part of the plan's element type - the dType<:Real
    case. Prefer `_rebuild` internally, which keeps the original `OT`/`trainable`. =#
function MonoConvFFT(σ, weight, bc, fftPlan, scale, averagingLayer;
    trainable = true, OT = nothing)
    D = ndims(weight) - 1
    p = fftPlan isa Tuple ? first(fftPlan) : fftPlan
    OT = OT === nothing ?
         (p === nothing ? Float32 : real(eltype(p))) :
         OT
    return MonoConvFFT{
        D,
        OT,
        typeof(σ),
        typeof(weight),
        typeof(bc),
        typeof(fftPlan),
        trainable,
        typeof(scale),
        typeof(averagingLayer)
    }(σ, weight, bc, fftPlan, scale, averagingLayer)
end

"""
    _rebuild(m::MonoConvFFT, weight, fftPlan)

Rebuild `m` with new `weight` and `fftPlan` (typically on a different device),
preserving every type parameter that is not recoverable from the fields -
notably `OT` and the `trainable` flag. This is the only accepted way for the
GPU extensions to reconstruct a layer.
"""
function _rebuild(m::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}, weight, fftPlan) where {D,OT,F,A,PD,P,T,S,AL}
    return MonoConvFFT{
        D,OT,F,typeof(weight),PD,typeof(fftPlan),T,S,AL
    }(m.σ, weight, m.bc, fftPlan, m.scale, m.averagingLayer)
end

# MonoConvFFT function
# (remark) Similar to ConvFFT function
# (remark) but omit bias b, Boundary, Analyticity An

function MonoConvFFT(w::AbstractArray{T,N}, originalSize, σ=identity; plan=true,
    boundary=FourierFilterFlux.Periodic(), dType=Float32, trainable=true,
    OT=Float32, scale, averagingLayer=false) where {T,N}

    @assert length(originalSize) >= N - 1
    if dType <: Complex
        OT = dType
    end

    if length(originalSize) == N - 1
        exSz = (originalSize..., 1) # default number of channels is 1
    else
        exSz = originalSize
    end

    netSize, boundary = effectiveSize(exSz[1:N-1], boundary)
    cDims = (1:(N-1)...,)   # renamed: `convDims` is now an accessor function

    if dType <: Complex && netSize[1] != size(w, 1)
        wtmp = ifftshift(ifft(w, cDims), cDims)
        wtmp = applyBC(wtmp, boundary, N - 1)
        w = fft(fftshift(wtmp, cDims), cDims)
    elseif dType <: Real && netSize[1] >> 1 + 1 != size(w, 1)
        wtmp = ifft(w, cDims)
        wtmp, _ = applyBC(wtmp, boundary, N - 1)
        w = fft(wtmp, cDims)
    end

    if plan
        # change to MonomakePlan
        fftPlan = MonomakePlan(dType, OT, w, exSz, boundary)
    else
        fftPlan = nothing
    end

    return MonoConvFFT{N - 1,OT,typeof(σ),typeof(w),typeof(boundary),typeof(fftPlan),trainable,typeof(scale),typeof(averagingLayer)}(σ, w, boundary, fftPlan, scale, averagingLayer)
end

# MonomakePlan function
# (remark) Similar to makePlan function

function MonomakePlan(dType, OT, w, exSz, boundary)
    N = ndims(w)
    netSize, boundary = effectiveSize(exSz[1:N-1], boundary)
    cDims = (1:(N-1)...,)
    nullEx = Adapt.adapt(typeof(w), zeros(dType, netSize..., exSz[N:end]...))
    if dType <: Real
        fftPlan = plan_fft(nullEx, cDims)
    else
        null2 = Adapt.adapt(typeof(w), zeros(dType, netSize..., exSz[N:end]...)) .+ 0im
        fftPlan = plan_fft(null2, cDims)
    end
end

function Base.show(io::IO, l::MonoConvFFT)
    # stored as a brief description
    sz = _planSize(l.fftPlan)
    es = originalSize(sz[1:ndims(l.weight)-1], l.bc)
    print(io, "MonoConvFFT[input=$(es), " *
              "nfilters = $(size(l.weight)[end]-1), " *
              "σ=$(l.σ), " *
              "bc=$(l.bc), " *
              "averagingLayer=$(l.averagingLayer)]")
end

include("MonogenicTransform.jl")
include("MonogenicConvFFTConstructors.jl")
export MonogenicLayer
export MonoFilterTypes, GaussianHP, GaussianLP

include("MonogenicUtils.jl");
export size, getBatchSize, outType, isTrainable, convDims

include("MonogenicParamCollection.jl")

include("MonogenicDevices.jl")
export monogenic_device, has_gpu

export MonoConvFFT

include("MonogenicOptional.jl")
export visual_first_layer, visual_second_layer
export rotate_image, inv_rotate_out

end
