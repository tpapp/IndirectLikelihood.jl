using Documenter
using IndirectLikelihood
using ContinuousTransformations

"""
Documentation generation wrapped into a single function for development.

Call `mkdocs()` repeatedly when working on the documentation interactively.
"""
mkdocs() = makedocs(format = :html,
                    sitename = "IndirectLikelihood",
                    pages = ["index.md",
                             "general.md",
                             "auxiliary.md",
                             "internals.md"])

mkdocs()

deploydocs(repo = "github.com/tpapp/IndirectLikelihood.jl.git",
           target = "build",
           deps = nothing,
           make = nothing,
           julia = "0.6")
