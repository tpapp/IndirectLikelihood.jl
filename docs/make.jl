using Documenter
using IndirectLikelihood
import ContinuousTransformations

makedocs(format = :html,
         sitename = "IndirectLikelihood",
         pages = ["index.md",
                  "general.md",
                  "internals.md"])
