# Identification data

All estimation methods in this package expect an object of type `AbstractIdData`, created using the function [`iddata`](@ref). This object typically holds input and output data as well as the sample time. 

```@docs
ControlSystemIdentification.iddata
ControlSystemIdentification.predictiondata
```

Some frequency-domain methods accept or return objects of type [`FRD`](@ref), representing frequency-response data. An `FRD` object can be created directly using the constructor, or using the appropriate [`iddata`](@ref) method above.


## Video tutorials

Relevant video tutorials are available here:


```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/QMO8cDpjw5U?si=MZuC1BwKoLtgGJ_y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/fP_ENSXURYA?si=2QcQSj-UUAkmjQYZ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```