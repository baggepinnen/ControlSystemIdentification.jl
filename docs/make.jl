ENV["GKSwstype"] = 100 # workaround for gr segfault on GH actions
using Documenter, ControlSystemIdentification, ControlSystemsBase, DelimitedFiles
# ENV["JULIA_DEBUG"]=Documenter # Enable this for debugging

using Plots
gr(fmt=:png)
using LeastSquaresOptim


# Run this code here to make sure the tutorial can access the file, not sure why it does not work from within the tutorial
using DelimitedFiles
url = "http://www.it.uu.se/research/publications/reports/2017-024/CoupledElectricDrivesDataSetAndReferenceModels.zip"
zipfilename = "/tmp/bd.zip"
cd("/tmp")
path = Base.download(url, zipfilename)
run(`unzip -o $path`)
data = readdlm("/tmp/DATAUNIF.csv", ',')[2:end, 1:4]




makedocs(
      sitename = "ControlSystemIdentification Documentation",
      doctest = false,
      modules = [ControlSystemIdentification, Base.get_extension(ControlSystemIdentification, :ControlSystemIdentificationLSOptExt)],
      pagesonly = true,
      draft = false,
      pages = [
            "Home" => "index.md",
            "Identification data" => "iddata.md",
            "State-space estimation" => "ss.md",
            "Transfer-function estimation" => "tf.md",
            "Impulse-response estimation" => "impulse.md",
            "Frequency-domain estimation" => "freq.md",
            "Validation" => "validation.md",
            "Nonlinear identification" => "nonlinear.md",
            "Examples" => [
                  "Temperature control" => "examples/temp.md",
                  "Identification in closed loop" => "examples/closed_loop_id.md",
                  "Identification of unstable systems" => "examples/unstable_systems.md",
                  "Delay estimation" => "examples/delayest.md",
                  "Ball and beam" => "examples/ballandbeam.md",
                  "Flexible robot arm" => "examples/flexible_robot.md",
                  "Glass furnace" => "examples/glass_furnace.md",
                  "Evaporator" => "examples/evaporator.md",
                  "Hair dryer" => "examples/hair_dryer.md",
                  "VARX model" => "examples/varx.md",
                  "Nonlinear belt drive" => "examples/hammerstein_wiener.md",
                  "Fit parameters of ModelingToolkit model" => "examples/modelingtoolkit.md",
            ],
            "API" => "api.md",
      ],
      warnonly = [:docs_block, :missing_docs],
      format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
)

deploydocs(
      repo = "github.com/baggepinnen/ControlSystemIdentification.jl.git",
)

cd(@__DIR__)