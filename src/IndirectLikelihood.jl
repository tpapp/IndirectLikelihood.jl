module IndirectLikelihood

using ArgCheck
using DocStringExtensions
using Parameters
using StatsBase

import Base: size, show

import StatsBase: loglikelihood


# general interface

"""
    MLE(data)

Return the maximum likelihood of the parameters using the given data, using a
model determined by its type. See also [`loglikelihood`](@ref).
"""
function MLE end

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
indirect_loglikelihood(y, x) = loglikelihood(y, MLE(x))


# model: multivariate normal

"""
    $SIGNATURES

Test if `matrix` is square and conformable with `vector`.
"""
function is_conformable_square(vector::AbstractVector, matrix::AbstractMatrix)
    k = length(vector)
    size(matrix) == (k, k)
end

"""
    MvNormal_SS(n, m, S)

Multivariate normal model summary statistics with `n` observations, mean `m` and
sample covariance `S`. Only saves the summary statistics.
"""
struct MvNormal_SS{TW <: Real, Tm <: AbstractVector, TS <: AbstractMatrix}
    "sum of the weights (alternatively, total number of observations)"
    W::TW
    "(weighted) sample mean"
    m::Tm
    "(weighted) sample covariance matrix"
    S::TS
    function MvNormal_SS(W::TW, m::Tm, S::TS) where {TW <: Real,
                                                    Tm <: AbstractVector,
                                                    TS <: AbstractMatrix}
        @argcheck is_conformable_square(m, S)
        new{TW, Tm, TS}(W, m, S)
    end
end

size(ss::MvNormal_SS) = ss.W, length(ss.m)

function show(io::IO, ss::MvNormal_SS)
    W, k = size(ss)
    print(io, "Summary statistics for multivariate normal, $(W) × $(k) samples")
end

"""
    $SIGNATURES

Multivariate normal summary statistics from observations (each row of `X` is an
observation).
"""
function MvNormal_SS(X::AbstractMatrix)
    m, S = mean_and_cov(X, 1; corrected = false)
    MvNormal_SS(size(X, 1), vec(m), S)
end

"""
    $SIGNATURES

Multivariate normal summary statistics from observations (each row of `X` is an
observation), with weights.
"""
function MvNormal_SS(X::AbstractMatrix, wv::AbstractWeights)
    m, S = mean_and_cov(X, wv, 1; corrected = false)
    MvNormal_SS(sum(wv), vec(m), S)
end

"""
    MvNormal_Params(μ, Σ)

Parameters for the multivariate normal model ``x ∼ MvNormal(μ, Σ)``.
"""
struct MvNormal_Params{Tμ, TΣ}
    "mean"
    μ::Tμ
    "variance"
    Σ::TΣ
    function MvNormal_Params(μ::Tμ, Σ::TΣ) where {Tμ, TΣ}
        @argcheck is_conformable_square(μ, Σ)
        new{Tμ, TΣ}(μ, Σ)
    end
end

MLE(ss::MvNormal_SS) = MvNormal_Params(ss.m, ss.S)

# NOTE: documenting this here, since the generic function is defined in another
# package
"""
    loglikelihood(data, params)

Return the log likelihood of the `data` under parameters `params`, which come
from the same model family.

`loglikelihood(data, MLE(data))` maximizes the loglikelihood for given `data`.
"""
function loglikelihood(ss::MvNormal_SS, params::MvNormal_Params)
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
    OLS_Data(Y, X)

Ordinary least squares with dependent variable `Y` and design matrix `X`.

Either

1. `Y` is an ``n×m`` matrix, then ``Y = X B + E`` where `X` is a ``n×k`` matrix,
`B` is a ``k×m`` parameter matrix, and ``Eᵢ ∼ N(0, Σ)`` is IID with ``m×m``
variance matrix ``Σ`` (multivariate linear regression), or

2. `Y` is a length `n` vector, then ``Y = X β + ϵ``, where `X` is a ``n×k``
matrix, `β` is a parameter vector of `k` elements, and `ϵᵢ ∼ N(0, σ)` where `σ`
is the variance of the normal error.

See [`OLS_Params`](@ref) for the parameters.
"""
struct OLS_Data{TY <: Union{AbstractMatrix, AbstractVector},
                TX <: AbstractMatrix}
    Y::TY
    X::TX
    function OLS_Data(Y::TY, X::TX) where {TY, TX}
        @argcheck size(Y, 1) == size(X, 1)
        new{TY, TX}(Y, X)
    end
end

function show(io::IO, data::OLS_Data{<:AbstractVector})
    n, k = size(data.X)
    print(io, "OLS regression, $(n) scalar observations, $(k) covariates")
end

function show(io::IO, data::OLS_Data{<:AbstractMatrix})
    n, k = size(data.X)
    m = size(data.Y, 2)
    print(io, "OLS regression, $(n) vector observations of length $(m), $(k) covariates")
end

"""
    OLS_Params(B, Σ)

Maximum likelihood estimated parameters for an OLS regression. See
[`OLS_Data`](@ref).
"""
struct OLS_Params{TB <: Union{AbstractMatrix, AbstractVector},
                  TΣ <: Union{<:Real, AbstractMatrix}}
    B::TB
    Σ::TΣ
    function OLS_Params(B::TB, Σ::TΣ) where {TB <: AbstractVector, TΣ <: Real}
        new{TB, TΣ}(B, Σ)
    end
    function OLS_Params(B::TB, Σ::TΣ) where {TB <: AbstractMatrix,
                                             TΣ <: AbstractMatrix}
        m = size(B, 2)
        @argcheck size(Σ) == (m, m)
        new{TB, TΣ}(B, Σ)
    end
end

function MLE(data::OLS_Data)
    @unpack Y, X = data
    B = qrfact(X) \ Y
    E = Y - X*B
    Σ = E'*E / size(X, 1)
    OLS_Params(B, Σ)
end

function loglikelihood(data::OLS_Data{TY}, params::OLS_Params{TB, TΣ}) where
    {TY <: AbstractVector, TB <: AbstractVector, TΣ <: Real}
    @unpack Y, X = data
    @unpack B, Σ = params
    n = length(Y)
    @argcheck size(X, 1) == n
    @argcheck size(X, 2) == length(B)
    E = Y - X * B
    -0.5 * (n*(log(2*π) + log(Σ)) + sum(abs2, E)/Σ)
end

function loglikelihood(data::OLS_Data{TY}, params::OLS_Params{TB, TΣ}) where
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
