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


# general interface

"""
    summary_statistics(::Type{T}, data...)

Compute summary statistics of the given type `T` using `data`. The
interpretation of `data` depends on `T`.
"""
function summary_statistics end

"""
    loglikelihood(summary_statistics, parameters)

Return the log likelihood of the data with the given `summary_statistics` under
and `parameters`, which come from the same model family.

`loglikelihood(ss, ML(ss))` maximizes the loglikelihood for given summary
statistics `ss`.
"""
function loglikelihood end

"""
    ML(summary_statistics)

Return the maximum likelihood of the parameters using the given
`summary_statistics`. See `loglikelihood`.
"""
function ML end

"""
    indirect_loglikelihood(y, x)

1. estimate the model from summary statistics `x` using maximum likelihood,

2. return the likelihood of summary_statistics `y` under the estimated
parameters.

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


# model: multivariate normal

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

function loglikelihood(ss::MvNormalSS, params::MvNormalParams)
    @unpack W, m, S = ss
    @unpack μ, Σ = params
    K = length(m)
    @argcheck length(μ) == K
    C = cholfact(Σ)
    d = m-μ
    -W/2*(K*log(2*π) + logdet(C) + dot(vec(S) + vec(d * d'), vec(inv(Σ))))
end


# OLS

"""
    OLS_data(Y, X)

Ordinary least squares with dependent variable `Y` and design matrix `X`.

Either

1. `Y` is an ``n×m`` matrix, then ``Y = X B + E`` where `X` is a ``n×k`` matrix,
`B` is a ``k×m`` parameter matrix, and ``Eᵢ ∼ N(0, Σ)`` is IID with ``m×m``
variance matrix ``Σ`` (multivariate linear regression), or

2. `Y` is a length `n` vector, then ``Y = X β + ϵ``, where `X` is a ``n×k``
matrix, `β` is a parameter vector of `k` elements, and `ϵᵢ ∼ N(0, σ)` where `σ`
is the variance of the normal error.

See [`OLS_params`](@ref) for the parameters.
"""
struct OLS_data{TY <: Union{AbstractMatrix, AbstractVector},
                TX <: AbstractMatrix}
    Y::TY
    X::TX
    function OLS_data(Y::TY, X::TX) where {TY, TX}
        @argcheck size(Y, 1) == size(X, 1)
        new{TY, TX}(Y, X)
    end
end

"""
    OLS_params(B, Σ)

Maximum likelihood estimated parameters for an OLS regression. See
[`OLS_data`](@ref).
"""
struct OLS_params{TB <: Union{AbstractMatrix, AbstractVector},
                  TΣ <: Union{<:Real, AbstractMatrix}}
    B::TB
    Σ::TΣ
    function OLS_params(B::TB, Σ::TΣ) where {TB <: AbstractVector, TΣ <: Real}
        new{TB, TΣ}(B, Σ)
    end
    function OLS_params(B::TB, Σ::TΣ) where {TB <: AbstractMatrix,
                                             TΣ <: AbstractMatrix}
        m = size(B, 2)
        @argcheck size(Σ) == (m, m)
        new{TB, TΣ}(B, Σ)
    end
end

function ML(data::OLS_data)
    @unpack Y, X = data
    B = qrfact(X) \ Y
    E = Y - X*B
    Σ = E'*E / size(X, 1)
    OLS_params(B, Σ)
end

function loglikelihood(data::OLS_data{TY}, params::OLS_params{TB, TΣ}) where
    {TY <: AbstractVector, TB <: AbstractVector, TΣ <: Real}
    @unpack Y, X = data
    @unpack B, Σ = params
    n = length(Y)
    @argcheck size(X, 1) == n
    @argcheck size(X, 2) == length(B)
    E = Y - X * B
    -0.5 * (n*(log(2*π) + log(Σ)) + sum(abs2, E)/Σ)
end

function loglikelihood(data::OLS_data{TY}, params::OLS_params{TB, TΣ}) where
    {TY <: AbstractMatrix, TB <: AbstractMatrix, TΣ <: AbstractMatrix}
    @unpack Y, X = data
    @unpack B, Σ = params
    n, m = size(Y)
    @argcheck size(B, 1) == size(X, 2) # k
    @argcheck size(B, 2) == m
    E = Y - X * B
    U = chol(Σ)
    A = E / U
    -0.5 * (n*(m*log(2*π) + 2*logdet(U)) + sum(abs2, A))
end

end # module
