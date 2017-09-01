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
    MvNormalSS

######################################################################
# general interface
######################################################################

"""
    ML(summary_statistics)

Return the maximum likelihood of the parameters as a tuple using the given
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
    indirect_loglikelihood(summary_statistics, data...)

1. estimate the model of `summary_statistics` on `data` using maximum
likelihood,

2. return the likelihood of `summary_statistics` under the estimated parameters.

Useful for indirect inference. See

- Gallant, A. R., & McCulloch, R. E. (2009). On the determination of general
  scientific models with application to asset pricing. Journal of the American
  Statistical Association, 104(485), 117–131.

- Drovandi, C. C., Pettitt, A. N., Lee, A., & others, (2015). Bayesian indirect
  inference using a parametric auxiliary model. Statistical Science, 30(1),
  72–95.
"""
function indirect_loglikelihood(ss::T, data...) where {T}
    loglikelihood(ss, ML(summary_statistics(T, data...))...)
end

######################################################################
# model: multivariate normal
######################################################################

"""
    MvNormalSS(n, m, S)

Multivariate normal model with `n` observations, mean `m` and sample covariance
`S`.
"""
struct MvNormalSS{Tm <: AbstractVector, TS <: AbstractMatrix}
    n::Int
    m::Tm
    S::TS
    function MvNormalSS(n::Int, m::Tm, S::TS) where {Tm <: AbstractVector,
                                                     TS <: AbstractMatrix}
        k = length(m)
        @argcheck size(S) == (k, k)
        new{Tm,TS}(n, m, S)
    end
end

size(ss::MvNormalSS) = ss.n, length(ss.m)

function show(io::IO, ss::MvNormalSS)
    n, k = size(ss)
    print(io, "Summary statistics for multivariate normal, $(n) × $(k) samples")
end

"""
    summary_statistics(::Type{MvNormalSS}, X)

Summary statistics for observations under a multivariate normal model. Each
observation is a row of `X`.
"""
function summary_statistics(::Type{<:MvNormalSS}, X::AbstractMatrix)
    n, k = size(X)
    m, S = mean_and_cov(X, 1; corrected = false)
    MvNormalSS(n, vec(m), S)
end

ML(ss::MvNormalSS) = ss.m, ss.S

"""
    loglikelihood(summary_statistics::MvNormalSS, μ, Σ)

Return the log likelihood of the data with the given `summary_statistics` and
parameters `μ` (mean) and `Σ` (variance matrix).

`loglikelihood(s, ML(s)...)` maximizes the loglikelihood for given summary
statistics `s`.
"""
function loglikelihood(ss::MvNormalSS, μ::AbstractVector, Σ::AbstractMatrix)
    @unpack n, m, S = ss
    K = length(m)
    @argcheck length(μ) == K && size(Σ) == (K, K)
    C = cholfact(Σ)
    d = m-μ
    -n/2*(K*log(2*π) + logdet(C) + dot(vec(S) + vec(d * d'), vec(inv(Σ))))
end

end # module
