using Documenter, ControlSystemIdentification

using Plots
plotly()


makedocs(
      sitename = "ControlSystemIdentification Documentation",
      doctest = false,
      modules = [ControlSystemIdentification],
      pages = [
            "Home" => "index.md",
            "Identification data" => "iddata.md",
            "State-space estimation" => "ss.md",
            "Transfer-function estimation" => "tf.md",
            "Impulse-response estimation" => "impulse.md",
            "Frequency-domain estimation" => "freq.md",
            "Validation" => "validation.md",
            "API" => "api.md",
      ],
      format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
) # Due to lots of plots, this will just have to be run on my local machine

deploydocs(
      deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-cinder"),
      repo = "github.com/baggepinnen/ControlSystemIdentification.jl.git",
)
