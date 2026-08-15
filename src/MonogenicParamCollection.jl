function Flux.trainable(m::MonoConvFFT{D,OT,F,A,PD,P,true,S,AL}) where {D,OT,F,A,PD,P,S,AL}
    (; weight = m.weight)
end

function Flux.trainable(::MonoConvFFT{D,OT,F,A,PD,P,false,S,AL}) where {D,OT,F,A,PD,P,S,AL}
    NamedTuple()
end
