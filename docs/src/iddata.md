# Identification data

All estimation methods in this package expect an object of type `AbstractIdData`, created using the function [`iddata`](@ref). This object typically holds input and output data as well as the sample time. 

```@docs
ControlSystemIdentification.iddata
ControlSystemIdentification.predictiondata
```

Some frequency-domain methods accept or return objects of type [`FRD`](@ref), representing frequency-response data. An `FRD` object can be created directly using the constructor, or using the appropriate [`iddata`](@ref) method above.