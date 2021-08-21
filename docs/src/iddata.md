# Identification data

All estimation methods in this package expects an object of type [`AbstractIdData`](@ref), created using the function [`iddata`](@ref). This object typically holds input and output data as well as the sample time. 

```@docs
ControlSystemIdentification.iddata
```


Some frequency-domain methods accept or return objects of type [`FRD`](@ref), representing frequency-response data
```@docs
ControlSystemIdentification.FRD
```