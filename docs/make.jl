ENV["GKSwstype"] = 322 # workaround for gr segfault on GH actions
using Documenter, ControlSystemIdentification, ControlSystemsBase, DelimitedFiles

using Plots

gr()


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
            "Examples" => [
                  "Temperature control" => "examples/temp.md"
                  "Identification in closed loop" => "examples/closed_loop_id.md"
                  "Ball and beam" => "examples/ballandbeam.md"
                  "Flexible robot arm" => "examples/flexible_robot.md"
                  "Glass furnace" => "examples/glass_furnace.md"
                  "Evaporator" => "examples/evaporator.md"
                  "Hair dryer" => "examples/hair_dryer.md"
            ],
            "API" => "api.md",
      ],
      strict = [:example_block],
      format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
) # Due to lots of plots, this will just have to be run on my local machine

deploydocs(
      # deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-cinder"),
      repo = "github.com/baggepinnen/ControlSystemIdentification.jl.git",
)
