module IndirectLikelihood

using ArgCheck
using Parameters
using StatsBase

import Base: size, show

import StatsBase: loglikelihood

export
    # general interface
    ML, summary_statistics, loglikelihood, indirect_loglikelihood,
    # specific models
    MvNormalSS, MvNormalParams

######################################################################
# general interface
######################################################################

"""
    ML(summary_statistics)

Return the maximum likelihood of the parameters using the given
`summary_statistics`. See `loglikelihood`.
"""
function ML end

"""
    summary_statistics(::Type{T}, data...)

Compute summary statistics of the given type `T` using `data`. The
interpretation of `data` depends on `T`.
"""
function summary_statistics end

"""
    indirect_loglikelihood(y, x)

1. estimate the model from summary statistics `x` using maximum likelihood,

2. return the likelihood of summary_statistics `y` under the estimated parameters.

Useful for pseudo-likelihood indirect inference, where `y` would be the observed
and `x` the simulated data. See in particular

- Gallant, A. R., & McCulloch, R. E. (2009). On the determination of general
  scientific models with application to asset pricing. Journal of the American
  Statistical Association, 104(485), 117–131.

- Drovandi, C. C., Pettitt, A. N., Lee, A., & others, (2015). Bayesian indirect
  inference using a parametric auxiliary model. Statistical Science, 30(1),
  72–95.
"""
indirect_loglikelihood(y, x) = loglikelihood(y, ML(x))

######################################################################
# model: multivariate normal
######################################################################

"""
    is_conformable_square(vector, matrix)

Test if `matrix` is square and conformable with `vector`.
"""
function is_conformable_square(vector::AbstractVector, matrix::AbstractMatrix)
    k = length(vector)
    size(matrix) == (k, k)
end

"""
    MvNormalSS(n, m, S)

Multivariate normal model with `n` observations, mean `m` and sample covariance
`S`.
"""
struct MvNormalSS{TW <: Real, Tm <: AbstractVector, TS <: AbstractMatrix}
    "sum of the weights (alternatively, total number of observations)"
    W::TW
    "(weighted) sample mean"
    m::Tm
    "(weighted) sample covariance matrix"
    S::TS
    function MvNormalSS(W::TW, m::Tm, S::TS) where {TW <: Real,
                                                    Tm <: AbstractVector,
                                                    TS <: AbstractMatrix}
        @argcheck is_conformable_square(m, S)
        new{TW, Tm, TS}(W, m, S)
    end
end

size(ss::MvNormalSS) = ss.W, length(ss.m)

function show(io::IO, ss::MvNormalSS)
    W, k = size(ss)
    print(io, "Summary statistics for multivariate normal, $(W) × $(k) samples")
end

"""
    summary_statistics(::Type{MvNormalSS}, X, [wv::AbstractWeights])

Summary statistics for observations under a multivariate normal model. Each
observation is a row of `X`. When `wv` is given, use it as weights.
"""
function summary_statistics(::Type{MvNormalSS}, X::AbstractMatrix)
    m, S = mean_and_cov(X, 1; corrected = false)
    MvNormalSS(size(X, 1), vec(m), S)
end

function summary_statistics(::Type{MvNormalSS},
                            X::AbstractMatrix, wv::AbstractWeights)
    m, S = mean_and_cov(X, wv, 1; corrected = false)
    MvNormalSS(sum(wv), vec(m), S)
end

"""
    MvNormalParams(μ, Σ)

Parameters for the multivariate normal model ``x ∼ MvNormal(μ, Σ)``.
"""
struct MvNormalParams{Tμ, TΣ}
    "mean"
    μ::Tμ
    "variance"
    Σ::TΣ
    function MvNormalParams(μ::Tμ, Σ::TΣ) where {Tμ, TΣ}
        @argcheck is_conformable_square(μ, Σ)
        new{Tμ, TΣ}(μ, Σ)
    end
end

ML(ss::MvNormalSS) = MvNormalParams(ss.m, ss.S)

"""
    loglikelihood(summary_statistics, parameters)

Return the log likelihood of the data with the given `summary_statistics` under
and `parameters`, which come from the same model family.

`loglikelihood(ss, ML(ss))` maximizes the loglikelihood for given summary
statistics `ss`.
"""
function loglikelihood(ss::MvNormalSS, params::MvNormalParams)
    @unpack W, m, S = ss
    @unpack μ, Σ = params
    K = length(m)
    @argcheck length(μ) == K
    C = cholfact(Σ)
    d = m-μ
    -W/2*(K*log(2*π) + logdet(C) + dot(vec(S) + vec(d * d'), vec(inv(Σ))))
end

end # module
