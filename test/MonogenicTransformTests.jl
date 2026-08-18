@testset "Monogenic transform (CPU)" begin

    @testset "output shape, 4-tensor input" begin
        @testset "scale=$scale" for scale in (1, 2, 4)
            sz = (16, 16, 1, 3)
            mono = MonogenicLayer(sz, scale=scale, σ=abs)
            x = randn(Float32, sz)
            y = mono(x)
            # 3 monogenic components per scale, plus the final averaging band
            @test size(y) == (16, 16, 3 * scale + 1, 1, 3)
            @test eltype(y) == Float32
            @test all(isfinite, y)
        end
    end

    @testset "output shape, 5-tensor input" begin
        sz = (16, 16, 4, 1, 3)
        mono = MonogenicLayer(sz, scale=2, σ=abs)
        y = mono(randn(Float32, sz))
        @test size(y) == (16, 16, 7, 4, 1, 3)
        @test eltype(y) == Float32
    end

    @testset "averagingLayer keeps only the last band" begin
        sz = (16, 16, 1, 3)
        full = MonogenicLayer(sz, scale=3, σ=abs, averagingLayer=false)
        avg = MonogenicLayer(sz, scale=3, σ=abs, averagingLayer=true)
        x = randn(Float32, sz)

        yf = full(x)
        ya = avg(x)
        @test size(ya) == (16, 16, 1, 1, 3)
        @test ya[:, :, 1, :, :] ≈ yf[:, :, end, :, :]
    end

    @testset "Float32 in, Float32 out" begin
        mono = MonogenicLayer((32, 32, 1, 2), scale=4, σ=abs)
        @test eltype(mono.weight) == ComplexF32
        y = mono(randn(Float32, 32, 32, 1, 2))
        @test eltype(y) == Float32
    end

    @testset "boundary conditions" begin
        sz = (16, 16, 1, 2)
        x = randn(Float32, sz)
        @testset "$(nameof(typeof(bc)))" for bc in (FourierFilterFlux.Periodic(),
            FourierFilterFlux.Sym(),
            FourierFilterFlux.Pad(5, 5))
            mono = MonogenicLayer(sz, scale=2, σ=abs, boundary=bc)
            y = mono(x)
            # the padding is stripped again, so the output is signal-sized
            @test size(y) == (16, 16, 7, 1, 2)
            @test eltype(y) == Float32
            @test all(isfinite, y)
        end
    end

    @testset "plan = false builds a plan per call" begin
        sz = (16, 16, 1, 2)
        x = randn(Float32, sz)
        planned = MonogenicLayer(sz, scale=2, σ=abs, plan=true)
        unplanned = MonogenicLayer(sz, scale=2, σ=abs, plan=false)
        @test unplanned.fftPlan === nothing
        @test unplanned(x) ≈ planned(x)
    end

    @testset "gradients" begin
        sz = (16, 16, 1, 2)
        mono = MonogenicLayer(sz, scale=2, σ=identity)
        x = randn(Float32, sz)

        ∇ = Zygote.gradient(t -> sum(abs2, mono(t)), x)[1]
        @test size(∇) == size(x)
        @test eltype(∇) == Float32
        @test all(isfinite, ∇)
        @test !all(iszero, ∇)

        #=  Spot-check against a finite difference. The transform is linear 
            in x before σ, so sum(abs2, ·) is a quadratic form and central
            differences are accurate. =#
        i = (8, 8, 1, 1)
        h = 1.0f-2
        xp = copy(x); xp[i...] += h
        xm = copy(x); xm[i...] -= h
        fd = (sum(abs2, mono(xp)) - sum(abs2, mono(xm))) / (2h)
        @test isapprox(fd, ∇[i...]; rtol=1.0f-2)
    end

end
