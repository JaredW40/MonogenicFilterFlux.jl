#=  Shared GPU-vs-CPU test body.

    CUDATests.jl and MetalTests.jl are the entry points (one per backend), but 
    the checks themselves live here and are parameterized by backend. =#

using BenchmarkTools
using LinearAlgebra: norm

# Relative error is done in the Frobenius norm. Preferred over elementwise `≈` 
# for the transform output, which is full of near-zero coefficients whose 
# relative error is meaningless, even when the array as a whole is correct.
_relerr(a, b) = norm(vec(a) .- vec(b)) / max(norm(vec(b)), eps(Float32))

function monogenic_gpu_suite(; name, toDevice, deviceArray, sync, allowscalar!)
    @testset "$name -- gpu()/cpu() converters" begin
        originalSize = (32, 32, 1, 4)
        m = MonogenicLayer(originalSize, scale=3, σ=abs,
            Monotype=GaussianLP(), averagingLayer=false)

        mg = toDevice(m)

        @test mg.weight isa deviceArray
        @test eltype(mg.weight) == eltype(m.weight)
        @test size(mg.weight) == size(m.weight)
        @test Array(mg.weight) ≈ m.weight

        # The plan must have been rebuilt, not copied: a device plan is no
        # longer an FFTW plan, but it describes the same buffer.
        @test !(mg.fftPlan isa FFTW.cFFTWPlan)
        @test mg.fftPlan isa AbstractFFTs.Plan
        @test size(mg.fftPlan) == size(m.fftPlan)
        @test eltype(mg.fftPlan) == eltype(m.fftPlan)

        #=  Type parameters that are not recoverable from the field values,
            and so are the ones a functor-walking reconstruction would lose.
            `trainable` in particular: MonogenicLayer defaults it to false, and
            a layer that comes back from the GPU claiming to be trainable would
            quietly start collecting gradients. =#
        @test outType(mg) == outType(m)
        @test isTrainable(mg) == isTrainable(m)
        @test ndims(mg) == ndims(m)
        @test typeof(mg.σ) == typeof(m.σ)
        @test mg.scale == m.scale
        @test mg.averagingLayer == m.averagingLayer
        @test isempty(Flux.trainable(mg))   # trainable = false by default

        # Round trip.
        m0 = cpu(mg)
        @test m0.weight isa Array
        @test m0.weight ≈ m.weight
        @test m0.fftPlan isa AbstractFFTs.Plan
        @test !(m0.fftPlan isa deviceArray)
        @test size(m0.fftPlan) == size(m.fftPlan)
        @test outType(m0) == outType(m)
        @test isTrainable(m0) == isTrainable(m)

        # Moving to a device you are already on is a no-op, not a rebuild.
        @test toDevice(mg) === mg
        @test cpu(m) === m

        # Derived accessors keep working once the plan is a device plan -
        # these used to read `fftPlan.sz`, which only FFTW plans have.
        @test size(mg) == size(m)
        @test getBatchSize(mg) == getBatchSize(m)
        @test sprint(show, mg) isa String
    end

    @testset "$name -- forward accuracy vs CPU" begin
        @testset "size=$sz scale=$scale avg=$avg" for
            sz in ((32, 32, 1, 2), (64, 64, 1, 4), (28, 28, 3, 2)),
            scale in (2, 4),
            avg in (false, true)

            m = MonogenicLayer(sz, scale=scale, σ=abs,
                Monotype=GaussianLP(), averagingLayer=avg)
            mg = toDevice(m)

            x = randn(Float32, sz)
            xg = toDevice(x)

            y = m(x)
            yg = mg(xg)
            sync()

            @test size(yg) == size(y)

            @test yg isa deviceArray

            @test eltype(y) == Float32
            @test eltype(yg) == Float32

            @test _relerr(Array(yg), y) < 1.0f-4
            @test isapprox(Array(yg), y; rtol=1.0f-3, atol=1.0f-5)
        end
    end

    @testset "$name -- higher rank input" begin
        # 5-tensor input, i.e. a second scattering layer
        sz = (32, 32, 4, 1, 2)
        m = MonogenicLayer(sz, scale=2, σ=abs, averagingLayer=false)
        mg = toDevice(m)

        x = randn(Float32, sz)
        y = m(x)
        yg = mg(toDevice(x))
        sync()

        @test yg isa deviceArray
        @test size(yg) == size(y)
        @test _relerr(Array(yg), y) < 1.0f-4
    end

    @testset "$name -- no scalar indexing" begin
        #=  Scalar indexing on a device array works, via a per-element host
            round trip that is thousands of times slower than the CPU version.
            It is a performance bug that produces correct numbers, so it can
            only be caught by forbidding it outright. =#
        m = MonogenicLayer((64, 64, 1, 2), scale=3, σ=abs)
        mg = toDevice(m)
        xg = toDevice(randn(Float32, 64, 64, 1, 2))

        allowscalar!(false)
        yg = mg(xg)
        sync()
        @test yg isa deviceArray
    end

    @testset "$name -- device mismatch is caught" begin
        m = MonogenicLayer((32, 32, 1, 2), scale=2, σ=abs)
        mg = toDevice(m)
        x = randn(Float32, 32, 32, 1, 2)

        @test_throws "same device" mg(x)
        @test_throws "same device" m(toDevice(x))
    end

    @testset "$name -- gradient parity" begin
        #=  σ = identity rather than the default abs: abs is not
            differentiable at zero and the transform output is 
            genuinely zero in places, which would make this test 
            about `abs` rather than about the backends. =#
        sz = (32, 32, 1, 2)
        m = MonogenicLayer(sz, scale=2, σ=identity, averagingLayer=false)
        mg = toDevice(m)

        x = randn(Float32, sz)
        xg = toDevice(x)

        ∇ = Zygote.gradient(t -> sum(abs2, m(t)), x)[1]
        ∇g = Zygote.gradient(t -> sum(abs2, mg(t)), xg)[1]
        sync()

        @test ∇g isa deviceArray
        @test size(∇g) == size(∇)
        @test eltype(∇g) == Float32
        @test _relerr(Array(∇g), ∇) < 1.0f-3
    end

    @testset "$name -- timing vs CPU" begin
        strict = get(ENV, "MONOGENIC_STRICT_PERF", "false") == "true"

        @testset "timing sz=$sz" for sz in ((64, 64, 1, 8), (128, 128, 1, 8), (256, 256, 1, 4))
            m = MonogenicLayer(sz, scale=4, σ=abs, averagingLayer=false)
            mg = toDevice(m)

            x = randn(Float32, sz)
            xg = toDevice(x)

            # warm up: first call pays plan construction and kernel compilation
            m(x)
            mg(xg)
            sync()

            t_cpu = @belapsed $m($x) samples = 10 seconds = 5
            t_gpu = @belapsed begin
                $mg($xg)
                $sync()
            end samples = 10 seconds = 5

            @info "MonoConvFFT $name timing" size = sz cpu_s = t_cpu gpu_s = t_gpu speedup = t_cpu / t_gpu

            if strict && prod(sz) >= 128 * 128
                @test t_gpu < t_cpu
            elseif t_gpu > t_cpu
                @warn "$name slower than the CPU at size $sz -- expected on small inputs or a shared runner, suspicious otherwise" cpu_s = t_cpu gpu_s = t_gpu
            end
        end
    end

    return nothing
end