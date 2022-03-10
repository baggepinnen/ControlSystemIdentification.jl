# Identification data

All estimation methods in this package expect an object of type [`AbstractIdData`](@ref), created using the function [`iddata`](@ref). This object typically holds input and output data as well as the sample time. 

```@docs
ControlSystemIdentification.iddata
ControlSystemIdentification.predictiondata
```

Some frequency-domain methods accept or return objects of type [`FRD`](@ref), representing frequency-response data.