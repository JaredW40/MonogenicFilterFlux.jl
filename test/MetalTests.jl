#=  Metal entry point.  =#

isdefined(@__MODULE__, :monogenic_gpu_suite) || include("gpuComparisons.jl")

if Metal.functional()

    @testset "Metal" begin

        @testset "Metal plan rebuilding" begin
            m = MonogenicLayer((32, 32, 1, 2), scale=3, σ=abs)

            @test m.fftPlan isa FFTW.cFFTWPlan

            mg = gpu(m)
            @test mg.fftPlan isa Metal.MtlFFTPlan
            @test mg.weight isa MtlArray

            mm = mtl(m)
            @test mm.fftPlan isa Metal.MtlFFTPlan
            @test typeof(mm) == typeof(mg)

            xg = MtlArray(randn(Float32, 32, 32, 1, 2))
            @test mg.fftPlan * ComplexF32.(xg) isa MtlArray

            back = cpu(mg)
            @test back.fftPlan isa FFTW.cFFTWPlan
            @test size(back.fftPlan) == size(m.fftPlan)
        end

        @testset "Float64 is rejected up front" begin
            #=  Apple GPUs have no Float64. Without this check the failure
                surfaces from inside a shader compile, with no indication that
                precision is the problem. =#
            m64 = MonogenicLayer((16, 16, 1, 2), scale=2, σ=abs, dType=Float64)
            @test real(eltype(m64.weight)) === Float64
            @test_throws "Float64" gpu(m64)
        end

        monogenic_gpu_suite(
            name = "Metal",
            toDevice = gpu,
            deviceArray = MtlArray,
            sync = Metal.synchronize,
            allowscalar! = Metal.allowscalar,
        )
    end

else
    @warn "Metal is loaded but not functional -- skipping the Metal test set"
end