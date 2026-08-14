#=  Parameter collection.

    Note the deliberate absence of `Flux.@layer MonoConvFFT` here. `@layer`
    expands to a `Functors.@functor`, which defines the same method that
    `Functors.@leaf MonoConvFFT` (in MonogenicFilterFlux.jl) defines, and 
    the later definition silently wins. Whichever way that race resolves, 
    one of two things breaks:

      - if `@functor` wins, the struct is walked field-by-field and rebuilt
        through the bare positional constructor, which cannot recover the `OT`
        or `trainable` type parameters -- so `gpu(MonogenicLayer(...))` quietly
        comes back `trainable = true` even though `MonogenicLayer` defaults to
        `false`;
      - if `@leaf` wins, `@layer` contributed nothing but a precompile-time
        "method definition overwritten" warning.

    So the layer stays a leaf and `trainable` is written out by hand. =#

function Flux.trainable(m::MonoConvFFT{D,OT,F,A,PD,P,true,S,AL}) where {D,OT,F,A,PD,P,S,AL}
    (; weight = m.weight)
end

function Flux.trainable(::MonoConvFFT{D,OT,F,A,PD,P,false,S,AL}) where {D,OT,F,A,PD,P,S,AL}
    NamedTuple()
end
