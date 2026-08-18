#=  Adapt, Functors and MLDataDevices are dependencies of MonogenicFilterFlux
    but not necessarily of whatever environment you are running from. Reaching
    them through the package rather than with `using` keeps this file working
    under a bare `include("test/runtests.jl")` from any environment that has
    the package at all, not just inside a Pkg.test sandbox. =#
const Functors = MonogenicFilterFlux.Functors
const adapt = MonogenicFilterFlux.adapt
const CPUDevice = MonogenicFilterFlux.CPUDevice

@testset "device movement machinery (no GPU required)" begin
    mono = MonogenicLayer((16, 16, 1, 2), scale=2, σ=abs)

    @testset "the layer is a Functors leaf" begin
        @test Functors.isleaf(mono)
    end

    @testset "cpu() is a no-op on a CPU layer" begin
        #=  Identity, not just equality: `adapt_structure(::CPUDevice, ...)`
            returns a layer already on the host unchanged, so no plan is
            rebuilt and no array is copied. If this ever weakens to `==`, the
            layer is being reconstructed on every `cpu()` call. =#
        @test cpu(mono) === mono
        @test cpu(cpu(mono)) === mono
    end

    @testset "gpu() behaves according to has_gpu()" begin
        gpu_available = has_gpu()
        @info "device movement tests" has_gpu = gpu_available device = monogenic_device()

        if gpu_available
            moved = gpu(mono)
            @test moved !== mono
            @test !(moved.weight isa Array)
            @test cpu(moved).weight ≈ mono.weight
  
            @test outType(moved) == outType(mono)
            @test isTrainable(moved) == isTrainable(mono)
        else
            @test gpu(mono) === mono
            @test monogenic_device() isa CPUDevice
        end
    end

    @testset "adapt(Array, ·) round trips" begin
        m2 = adapt(Array, mono.weight)
        @test m2 isa Array
        @test m2 ≈ mono.weight
    end

    @testset "device mismatch check" begin
        # Same-device is silent...
        @test MonogenicFilterFlux._checkMatchingDevice(randn(Float32, 16, 16, 1, 2), mono) === nothing
        # ...and the backend accessor is what the message is built from.
        @test MonogenicFilterFlux._backend(randn(2, 2)) === Array
    end

    @testset "plan geometry accessors are backend independent" begin
        #= `size(plan)` rather than `plan.sz`: only FFTW plans have that field,
           so every accessor that used it broke as soon as the layer moved. =#
        @test MonogenicFilterFlux._planSize(mono.fftPlan) == (16, 16, 1, 2)
        @test MonogenicFilterFlux._planEltype(mono.fftPlan) == ComplexF32
        @test MonogenicFilterFlux._planSize(nothing) == ()

        # The CPU plan builder is shared by both extensions' CPUDevice paths.
        p = MonogenicFilterFlux._cpuPlan(mono)
        @test p isa AbstractFFTs.Plan
        @test size(p) == size(mono.fftPlan)
        @test eltype(p) == eltype(mono.fftPlan)
    end
end