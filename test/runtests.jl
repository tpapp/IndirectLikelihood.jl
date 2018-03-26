using IndirectLikelihood

using IndirectLikelihood:
    # imported for testing
    vec_parameters

import IndirectLikelihood:
    # problem API
    simulate_data, common_random, common_random!, MLE, loglikelihood

using Base.Test

using ContinuousTransformations
using Distributions: logpdf, Normal, MvNormal
using Optim
using StatsBase: Weights
using Suppressor

const RNG = Base.Random.GLOBAL_RNG

"Extract first line."
firstline(lines) = split(lines, '\n')[1]

include("utilities.jl")
include("mvnormal.jl")
include("ols.jl")
include("general.jl")
include("../docs/make.jl")
