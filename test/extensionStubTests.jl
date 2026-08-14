@testset "optional extensions" begin

    #=  Plotting and image rotation moved into extensions. The names must stay
        available either way, as `using MonogenicFilterFlux` should not start
        throwing UndefVarError for code written before the split. =#
    @test visual_first_layer isa Function
    @test visual_second_layer isa Function
    @test rotate_image isa Function
    @test inv_rotate_out isa Function

    if Base.get_extension(MonogenicFilterFlux, :PyPlotExt) === nothing
        @test_throws "using PyPlot" visual_first_layer(zeros(Float32, 4, 4, 3, 1))
        @test_throws "using PyPlot" visual_second_layer(zeros(Float32, 4, 4, 3, 3, 1))
    else
        @info "PyPlotExt is loaded here -- the fallback error path is not exercised"
    end

    if Base.get_extension(MonogenicFilterFlux, :ImageRotationExt) === nothing
        @test_throws "using ImageTransformations" rotate_image(zeros(Float32, 4, 4, 1, 1), 0.1)
        @test_throws "using ImageTransformations" inv_rotate_out(zeros(Float32, 4, 4, 1, 1),
            zeros(Float32, 4, 4), 0.1)
    else
        @info "ImageRotationExt is loaded here -- the fallback error path is not exercised"
    end
end
