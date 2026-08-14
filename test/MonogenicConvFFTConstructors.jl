@testset "Monogenic ConvFFT constructors" begin

    @testset "random initialization" begin
        originalSize = (20, 16, 1, 10)
        mono = MonogenicLayer(originalSize, scale=4, σ=abs,
            Monotype=GaussianLP(), averagingLayer=false)
        @test mono.σ == abs
        @test mono.scale == 4
        @test mono.averagingLayer == false
        @test ndims(mono) == 2
        @test outType(mono) == Float32
        @test isTrainable(mono) == false # MonogenicLayer default
        @test isempty(Flux.trainable(mono))
        @test convDims(mono) == (1, 2)

        # Riesz kernel, then (RHO, HP, LP) per scale
        @test size(mono.weight) == (20, 16, 1 + 3 * 4)
        @test eltype(mono.weight) == ComplexF32
        @test mono.fftPlan isa AbstractFFTs.Plan
        @test size(mono.fftPlan) == originalSize
    end

    @testset "filter types" begin
        for Monotype in (GaussianLP(), GaussianHP())
            mono = MonogenicLayer((16, 16, 1, 2), scale=2, Monotype=Monotype)
            w = mono.weight
            #=  LP and HP are built to be complementary: |LP|² + |HP|² == 1 at
                every frequency. If that stops holding the filter bank is no
                longer a partition of the spectrum, whatever else still works. =#
            for j in 1:2
                HP = abs.(w[:, :, 3+3*(j-1)])
                LP = abs.(w[:, :, 4+3*(j-1)])
                @test all(isapprox.(HP .^ 2 .+ LP .^ 2, 1.0f0; atol=1.0f-4))
            end
        end
    end

    @testset "trainable = true" begin
        mono = MonogenicLayer((16, 16, 1, 2), scale=2, trainable=true)
        @test isTrainable(mono) == true
        trn = Flux.trainable(mono)
        @test haskey(trn, :weight)
        @test trn.weight === mono.weight
    end

    @testset "dType = Float64" begin
        mono = MonogenicLayer((16, 16, 1, 2), scale=2, dType=Float64)
        @test eltype(mono.weight) == ComplexF64
        @test outType(mono) == Float64
    end

    @testset "bare constructor" begin
        #=  The positional constructor exists so that any Functors-style
            reconstruction has a method to call; without one, reconstruction
            fails with a bare MethodError. Field order must match the struct. =#
        mono = MonogenicLayer((16, 16, 1, 2), scale=2)
        rebuilt = MonoConvFFT(mono.σ, mono.weight, mono.bc, mono.fftPlan,
            mono.scale, mono.averagingLayer)
        @test rebuilt.weight === mono.weight
        @test ndims(rebuilt) == ndims(mono)
        @test outType(rebuilt) == outType(mono)
    end

    @testset "accessors survive" begin
        mono = MonogenicLayer((20, 16, 1, 10), scale=2)
        @test getBatchSize(mono) == 10
        # `size` used to index the plan with ndims(weight[1]) == 0 and so
        # dropped the signal dimensions entirely.
        @test size(mono) == (20, 16, 1, 10)
        @test sprint(show, mono) isa String
    end
end
