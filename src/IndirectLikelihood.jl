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
import StatsBase: loglikelihood

include("utilities.jl")
include("general.jl")
include("mvnormal.jl")
include("ols.jl")

end # module
