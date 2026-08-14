#=  Boundary handling for the monogenic layer. =#
function applyBC(x, bc::FourierFilterFlux.Periodic, nd)
    return (x, axes(x)[1:nd])
end

function applyBC(x, bc::Sym, nd)
    flipThisDim = cat(x, reverse(x, dims=nd), dims=nd)
    if nd == 1
        return flipThisDim, axes(x)[1:nd]
    else
        return applyBC(flipThisDim, bc, nd - 1)[1], axes(x)[1:nd]
    end
end

#=  Zero padding. =#
function applyBC(x, bc::FourierFilterFlux.Pad, nd)
    usedInds = ntuple(ii -> bc.padBy[ii] .+ (1:size(x, ii)), ndims(bc))
    return (FourierFilterFlux.pad(x, bc.padBy), usedInds)
end
