__precompile__()
module IndirectLikelihood

import Base: size, show

using ArgCheck: @argcheck
using ContinuousTransformations: transform, inverse
using DocStringExtensions: SIGNATURES
using ForwardDiff: jacobian
using MacroTools
using Parameters: @unpack
using StatsBase: mean_and_cov, AbstractWeights

"""
The default random number generator for methods in this package, used when not
specified in the first argument.
"""
const RNG = Base.Random.GLOBAL_RNG

include("utilities.jl")
include("general.jl")
include("mvnormal.jl")
include("ols.jl")

end # module
