using Documenter
using IndirectLikelihood

"""
Documentation generation wrapped into a single function for development.

Call `mkdocs()` repeatedly when working on the documentation interactively.
"""
mkdocs() = makedocs(format = :html,
                    sitename = "IndirectLikelihood",
                    pages = ["index.md",
                             "general.md",
                             "internals.md"])

mkdocs()
