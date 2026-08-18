import FourierFilterFlux: getBatchSize, originalSize, originalDomain
import Base: size

#=  Plan geometry helpers.

    `size(::AbstractFFTs.Plan)` rather than `plan.sz`. A cuFFT plan stores 
    `input_size` and a Metal plan something else again, so anything touching 
    `.sz` breaks the moment the layer moves onto a GPU. Tuple-wrapped plans 
    unwrap to element 1, matching FourierFilterFlux's `_unwrap_plan`. =#

_planSize(p::Nothing) = ()
_planSize(p::Tuple) = _planSize(first(p))
_planSize(p) = size(p)

_planEltype(p::Nothing) = ComplexF32
_planEltype(p::Tuple) = _planEltype(first(p))
_planEltype(p) = eltype(p)

"""
    getBatchSize(c::MonoConvFFT)

Number of examples the layer's plan was built for.
"""
getBatchSize(c::MonoConvFFT) = _planSize(c.fftPlan)[end]

#=  ConvFFT stores a tuple of filter banks, so `weight[1]` is an array there and
    `ndims(weight[1])` is its convolved-dimension count. MonoConvFFT stores a
    single 3-tensor, so `weight[1]` is a scalar, `ndims` of it is 0, and
    `sz[1:0]` silently dropped the signal dimensions - which happens to give
    the right answer for Periodic and the padded size for everything else. The
    convolved-dimension count is the layer's own `D` parameter. =#
function size(l::MonoConvFFT)
    D = ndims(l)
    sz = _planSize(l.fftPlan)
    signalSize = originalSize(sz[1:D], l.bc)
    return (signalSize..., sz[(D+1):end]...)
end

"""
    originalDomain(m::MonoConvFFT; σ = identity)

The monogenic filter bank pulled back to the space domain, on the CPU whatever
device the layer is on, with `σ` applied pointwise.
"""
function originalDomain(m::MonoConvFFT; σ=identity)
    w = adapt(Array, m.weight)
    return σ.(ifft(w, convDims(m)))
end