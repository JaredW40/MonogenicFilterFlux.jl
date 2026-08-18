#=  CUDA entry point.

    Loaded only when runtests.jl's GROUP selection allows CUDA and the CUDA stack 
    actually resolved. `CUDA.functional()` is the last of the three layers. =#

isdefined(@__MODULE__, :monogenic_gpu_suite) || include("gpuComparisons.jl")

if CUDA.functional()
    @testset "CUDA" begin
        # Backend-specific plan and mover checks; everything else is the
        # shared suite below.
        @testset "cuFFT plan rebuilding" begin
            m = MonogenicLayer((32, 32, 1, 2), scale=3, σ=abs)

            @test m.fftPlan isa FFTW.cFFTWPlan

            mg = gpu(m)
            @test mg.fftPlan isa cuFFT.CuFFTPlan
            @test mg.weight isa CuArray

            # `cu(layer)` was removed when the plan could not be moved; it is
            # back, and must agree with `gpu`.
            mc = cu(m)
            @test mc.fftPlan isa cuFFT.CuFFTPlan
            @test typeof(mc) == typeof(mg)

            # The plan really is usable, not just correctly typed.
            xg = CuArray(randn(Float32, 32, 32, 1, 2))
            @test mg.fftPlan * ComplexF32.(xg) isa CuArray

            back = cpu(mg)
            @test back.fftPlan isa FFTW.cFFTWPlan
            @test size(back.fftPlan) == size(m.fftPlan)
            @test eltype(back.fftPlan) == eltype(m.fftPlan)
        end

        monogenic_gpu_suite(
            name = "CUDA",
            toDevice = gpu,
            deviceArray = CuArray,
            sync = CUDA.synchronize,
            allowscalar! = CUDA.allowscalar,
        )
    end

else
    @warn "CUDA is loaded but not functional -- skipping the CUDA test set"
end