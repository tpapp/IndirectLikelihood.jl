module IndirectLikelihood

using ArgCheck
using Parameters
using StatsBase

import Base: size, show

import StatsBase: loglikelihood

export ML, loglikelihood, MvNormalSS

"""
    ML(summary_statistics)

Return the maximum likelihood of the parameters as a tuple using the given
`summary_statistics`. See `loglikelihood`.
"""
function ML end

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

"""
    MvNormalSS(X)

Summary statistics for observations under a multivariate normal model. Each
observation is a row of `X`.
"""
function MvNormalSS(X::AbstractMatrix)
    n, k = size(X)
    m, S = mean_and_cov(X, 1; corrected = false)
    MvNormalSS(n, vec(m), S)
end

ML(ss::MvNormalSS) = ss.m, ss.S

size(ss::MvNormalSS) = ss.n, length(ss.m)

function show(io::IO, ss::MvNormalSS)
    n, k = size(ss)
    print(io, "Summary statistics for multivariate normal, $(n) × $(k) samples")
end

"""
    loglikelihood(summary_statistics, parameters...)

Return the log likelihood of the data with the given `summary_statistics` and
`parameters` for the relevant model (determined by the type of the first
argument).

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
