#=  The monogenic forward transform.

    This file used to hold three near-identical copies of the same routine
    (one for 4-tensors, one for 5-tensors, one generic). They computed the same
    thing - for N = 4 the generic permutation `(1,2,N+1,3,...,N)` evaluates to
    `(1,2,5,3,4)` and for N = 5 to `(1,2,6,3,4,5)`, exactly the two hard-coded
    tuples - so they have been collapsed into the single generic method below.
    Three copies of a routine that has to be correct on three backends is three
    places for the backends to drift apart.

    Two changes here allow the GPU path to work, and both are about element 
    types rather than about kernels:

    1. `_matchPlanEltype`. Applying a plan to an array whose eltype differs
       from the plan's sends AbstractFFTs down its `copy1` fallback, which
       allocates a host `Array` - on a GPU that either silently drags the
       whole pipeline back to the CPU or errors out. The plan is always
       complex (`plan_fft` complexifies real input), while `lfband` is real 
       at the top of every scale iteration, so this conversion happens on 
       every single pass and has to stay on-device.

    2. The normalization constant. `norm_csts[scal] * sqrt(2)` was `Float64`,
       so `LF .* w ./ (...)` promoted the entire pipeline from ComplexF32 to
       ComplexF64, which again lands in `copy1`, and which Metal cannot
       represent at all (no Float64 support in the hardware). It is now built
       in the filters' own real precision.

    Gradients go through AbstractFFTs' own rrules for `*` and `\`. Unlike
    FourierFilterFlux, nothing here uses a real-input `rfft` plan, as these are
    all complex `plan_fft`s, which use the plain `FFTAdjointStyle` that every
    backend implements correctly. That is why this file needs no hand-written
    adjoints, and why the extensions are as small as they are. =#

# Fast path: eltypes already agree, hand the array straight to the plan.
_matchPlanEltype(plan, x) = _matchPlanEltype(eltype(plan), x)
_matchPlanEltype(::Type{T}, x::AbstractArray{T}) where {T} = x
_matchPlanEltype(::Type{Complex{R}}, x::AbstractArray{R}) where {R<:Real} = complex.(x)
_matchPlanEltype(::Type{T}, x::AbstractArray) where {T} = T.(x)

#=  `p \ y` expands to `inv(p) * y`, and `inv` is memoised on the plan itself:
    AbstractFFTs requires every Plan to carry a `pinv` field and fills it in on
    first use, so the inverse plan is built once per plan rather than once per
    call, on every backend. That is why there is no plan cache here despite the
    transform inverting three times per scale.

    The two one-line indirections below exist so that a backend which turns out
    not to memoise can be given a cache in its extension without touching the
    numerics, which is the shape FourierFilterFlux's CUDAExt ended up needing
    for its from-scratch `plan_ifft` calls. =#
_monoFFT(plan, x) = plan * _matchPlanEltype(plan, x)
_monoIFFT(plan, y) = plan \ _matchPlanEltype(plan, y)

#=  With `plan = false` the layer carries no plan and one is built per call,
    on whatever device the (already boundary-corrected) signal lives on -
    `plan_fft` dispatches on the array type, so this yields an FFTW plan for an
    Array and a cuFFT/Metal plan for a device array with no branching here. =#
_ensurePlan(::Nothing, xbc, region) =
    Zygote.ignore_derivatives() do
        plan_fft(_matchPlanEltype(Complex{real(eltype(xbc))}, xbc), region)
    end
_ensurePlan(p::Tuple, xbc, region) = first(p)
_ensurePlan(p, xbc, region) = p

"""
    (Mono::MonoConvFFT)(x)

Apply the monogenic filter bank to `x`.

    input:  (nx, ny, extra dims..., nchannels, nexamples)
    output: (nx, ny, nfilters, extra dims..., nchannels, nexamples)

The filter bank is stored as `w[:, :, k]` with the Riesz kernel at `k = 1` and
then, for each scale `j`, the radial coordinate, high-pass and low-pass at
`k = 2 + 3(j-1)`, `3 + 3(j-1)` and `4 + 3(j-1)`.

`x` must be on the same device as the layer; see `gpu`/`cpu`.
"""
function (Mono::MonoConvFFT)(x::AbstractArray{<:Number})
    N = ndims(x)
    Zygote.ignore_derivatives() do
        _checkMatchingDevice(x, Mono)
    end

    scale = Mono.scale
    averagingLayer = Mono.averagingLayer
    w = Mono.weight

    # First filter: Rz
    # Second filter: RHO
    # Third filter: HP
    # Fourth filter: LP

    #=  Cancel the normalization so the frequency responses live in [0;1].
        Built in the filters' real precision (Float32 for the default layer). =#
    RT = real(eltype(w))
    norm_csts = RT(2) .^ (0:scale-1)
    sqrt2 = sqrt(RT(2))

    xbc, usedInds = applyBC(x, Mono.bc, ndims(w) - 1)
    fftPlan = _ensurePlan(Mono.fftPlan, xbc, convDims(Mono))
    if size(xbc) != size(fftPlan)
        xbc = reshape(xbc, size(fftPlan))
    end

    lfband = xbc
    local nextLayer
    local avg_out

    for scal = 1:scale
        LF = _monoFFT(fftPlan, lfband)
        lfband = real.(_monoIFFT(fftPlan, LF .* w[:, :, 4+3*(scal-1)]))
        HF = LF .* w[:, :, 3+3*(scal-1)] ./ (norm_csts[scal] * sqrt2)

        riez = _monoIFFT(fftPlan, HF .* w[:, :, 1])   # Riesz transform

        out1 = real.(_monoIFFT(fftPlan, HF))  # Primary part
        out2 = real.(riez)                    # x-Riesz part
        out3 = imag.(riez)                    # y-Riesz part

        # get rid of the padding
        out1 = out1[usedInds..., axes(out1)[length(usedInds)+1:end]...]
        out2 = out2[usedInds..., axes(out2)[length(usedInds)+1:end]...]
        out3 = out3[usedInds..., axes(out3)[length(usedInds)+1:end]...]

        if scal == 1
            nextLayer = cat(out1, out2, out3, dims=N + 1)
            # average layer filtering
            avg_out = lfband[usedInds..., axes(lfband)[length(usedInds)+1:end]...]
        else
            nextLayer = cat(nextLayer, out1, out2, out3, dims=N + 1)
        end

        if scal == scale
            nextLayer = cat(nextLayer, avg_out, dims=N + 1)
            if averagingLayer == true
                # nextLayer has N+1 dimensions here, so N colons then the last slice
                idx_slice = ntuple(_ -> Colon(), N)
                nextLayer = nextLayer[idx_slice..., end:end]
            end
        end

        lfband = lfband .* 2  # normalization due to the undecimated setting
    end

    #=  Move the filter axis into position 3: (1, 2, N+1, 3, 4, ..., N).
        Built as a tuple rather than by mutating a Vector - Zygote refuses to
        differentiate through `setindex!`, and now that the 4- and 5-tensor
        methods are gone every gradient call comes through here. =#
    perm = (1, 2, N + 1, ntuple(i -> i + 2, N - 2)...)

    nextLayer = permutedims(nextLayer, perm)

    return Mono.σ.(nextLayer)
end
