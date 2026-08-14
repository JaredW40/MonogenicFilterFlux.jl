#=  Optional functionality, provided by extensions.

    Four functions live outside the core package now: the two plotting routines
    (ext/PyPlotExt.jl, triggered by PyPlot) and the two image rotation helpers
    (ext/ImageRotationExt.jl, triggered by ImageTransformations). Neither
    dependency is needed to build or apply a monogenic layer, and both are
    expensive to require - PyPlot builds a Python/matplotlib stack, and the
    Images stack is large - so they are the caller's choice now.

    Declaring the names here rather than only in the extensions matters for two
    reasons. `MonogenicFilterFlux.visual_first_layer` stays a valid, exportable
    binding whether or not PyPlot is loaded, so `using MonogenicFilterFlux`
    behaves the same either way. And the fallback methods below turn "you
    forgot to load PyPlot" into a sentence that says so, instead of an
    UndefVarError that reads as though the function does not exist at all. =#

function visual_first_layer end
function visual_second_layer end
function rotate_image end
function inv_rotate_out end

function _extensionMissing(fname, pkg)
    error("MonogenicFilterFlux.$fname is provided by an extension: run " *
          "`using $pkg` (and add it to your project if necessary) to enable it.")
end

#=  Vararg fallbacks. The extensions define two-argument methods, which are
    more specific than `args...`, so loading an extension shadows these
    without any explicit precedence handling. =#
visual_first_layer(args...; kwargs...) = _extensionMissing("visual_first_layer", "PyPlot")
visual_second_layer(args...; kwargs...) = _extensionMissing("visual_second_layer", "PyPlot")
rotate_image(args...; kwargs...) = _extensionMissing("rotate_image", "ImageTransformations")
inv_rotate_out(args...; kwargs...) = _extensionMissing("inv_rotate_out", "ImageTransformations")
