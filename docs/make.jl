using Documenter
using IndirectLikelihood
import ContinuousTransformations

"""
Documentation generation wrapped into a single function for development.

Run this file as

```sh
julia -i make.jl
```

and call `mkdocs()` repeatedly when working on the documentation.
"""
mkdocs() = makedocs(format = :html,
                    sitename = "IndirectLikelihood",
                    pages = ["index.md",
                             "general.md",
                             "internals.md"])

mkdocs()
