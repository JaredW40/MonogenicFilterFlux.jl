#=  Device movement for MonoConvFFT.

    The hard part of putting a MonoConvFFT on a GPU is that `fftPlan` is not
    data. An FFTW plan is a handle to a plan FFTW built for a specific buffer
    shape on the host; a cuFFT/Metal plan is a handle to backend state on the
    device. Neither survives a device copy - `Adapt`ing the *fields* of a plan
    produces an object that looks like a plan, but segfaults or errors the
    moment it is applied. Every plan must therefore be *rebuilt from its
    geometry* on the target device, which is exactly what ConvFFT does in
    FourierFilterFlux. 

    A plan's geometry, for our purposes, is three things:

        size(plan)      the padded buffer shape (backend-independent accessor)
        eltype(plan)    always a Complex type -- see below
        convDims(m)     (1, ..., D), the transformed dimensions

    `convDims` comes off the layer's type parameter rather than the plan,
    because the transformed region is *not* recoverable from a plan in a
    backend-independent way (FFTW exposes `.region`, other backends do not).

    On element type: `MonomakePlan` calls `plan_fft` on a real array in the
    `dType <: Real` branch, and AbstractFFTs promotes that to a complex plan
    (`plan_fft(::AbstractArray{<:Real})` complexifies first). So a plan's
    eltype is always complex, in both branches, and `complex(eltype(p))` below
    is defensive idempotence rather than a real conversion. =#

"""
    _backend(x)

The array constructor behind `x`, e.g. `Array`, `CuArray`, `MtlArray`. Used
only for producing legible device-mismatch errors.
"""
_backend(x::AbstractArray) = Base.typename(typeof(x)).wrapper
_backend(::Nothing) = Nothing

"""
    _checkMatchingDevice(x, m::MonoConvFFT)

Fail fast, and legibly, when the signal and the filter bank live on different
devices. Without this the failure surfaces as an opaque GPU kernel compilation
error (or, worse, a silent scalar-indexing fallback that is thousands of times
slower than the CPU path).
"""
function _checkMatchingDevice(x::AbstractArray, m::MonoConvFFT)
    _backend(x) === _backend(m.weight) || error(
        "MonoConvFFT: the signal is a $(_backend(x)) but the filter bank is a " *
        "$(_backend(m.weight)) -- move both onto the same device first, e.g. " *
        "`gpu(layer)(gpu(x))` or `cpu(layer)(cpu(x))`.")
    return nothing
end

"""
    _cpuPlan(m::MonoConvFFT)

Rebuild `m`'s fft plan on the host. Used by the `CPUDevice` side of every
extension, so that the CPU-plan construction lives in exactly one place.
"""
_cpuPlan(m::MonoConvFFT) = _cpuPlan(m.fftPlan, convDims(m))
_cpuPlan(p::Nothing, region) = nothing
_cpuPlan(p::Tuple, region) = map(pi -> _cpuPlan(pi, region), p)
function _cpuPlan(p, region)
    plan_fft(zeros(complex(eltype(p)), size(p)...), region)
end

#=  Already on the host: nothing to rebuild. This is separate from Adapt's
    `adapt_structure(to, x) = x` fallback only so that the intent is explicit
    and so `cpu(cpu(m))` is provably free. The `A<:Array` bound also keeps this
    method from colliding with the `A<:CuArray` / `A<:MtlArray` methods the two
    extensions define for this same `::CPUDevice` first argument. =#
function Adapt.adapt_structure(::CPUDevice,
    m::MonoConvFFT{D,OT,F,A,PD,P,T,S,AL}) where {D,OT,F,A<:Array,PD,P,T,S,AL}
    return m
end

function _gpuDevice()
    dev = MLDataDevices.gpu_device()
    dev isa CPUDevice || return dev
    if isdefined(MLDataDevices, :reset_gpu_device!)
        MLDataDevices.reset_gpu_device!()
        dev = MLDataDevices.gpu_device()
    end
    return dev
end

"""
    has_gpu()

Whether a functional GPU backend (CUDA or Metal) is loaded and usable. Loading
neither package is not an error - it just means this returns `false` and the
layers stay on the CPU.
"""
has_gpu() = !(_gpuDevice() isa CPUDevice)

"""
    monogenic_device()

The device `gpu`/`cpu` will move a `MonoConvFFT` onto: a functional GPU device
if one is available, `CPUDevice()` otherwise. 
"""
monogenic_device() = MLDataDevices.gpu_device()
