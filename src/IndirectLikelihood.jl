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

export
    # general interface
    MLE, indirect_loglikelihood, loglikelihood,
    # modeling API
    IndirectLikelihoodProblem, simulate_problem, local_jacobian, simulate_data,
    random_crn, random_crn!,
    # specific auxiliary models and utilities
    MvNormalModel, OLSModel, add_intercept


# utilities

"""
    $SIGNATURES

Informative error message for missing method.
"""
macro no_method_info(ex)
    @capture(ex, f_(args__)) || error("Expected a function name with arguments")
    msg = "You need to define `$(string(f))` with this model type."
    f = esc(f)
    args = map(esc, args)
    quote
        function $f($(args...))
            info($msg)
            MethodError($f, ($(args...)))
        end
    end
end


# general interface

"""
    $SIGNATURES

Return the maximum likelihood of the parameters for `data` in `model`. See also
the 3-parameter version of [`loglikelihood`](@ref) in this package.
"""
@no_method_info MLE(model, data)

"""
    $SIGNATURES

Log likelihood of `data` under `model` with `parameters`.
"""
@no_method_info loglikelihood(model, data, parameters)

"""
    indirect_loglikelihood(model, y, x)

1. estimate `model` from summary statistics `x` using maximum likelihood,

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
indirect_loglikelihood(model, y, x) = loglikelihood(model, y, MLE(model, x))


# model: multivariate normal

"""
Model observations as drawn from a multivariate normal distribution. See
`MvNormalData` for summarizing data for estimation and likelihood calculations.
"""
struct MvNormalModel end

"""
    $SIGNATURES

Test if `matrix` is square and conformable with `vector`.
"""
function is_conformable_square(vector::AbstractVector, matrix::AbstractMatrix)
    k = length(vector)
    size(matrix) == (k, k)
end

"""
    MvNormalData(n, m, S)

Multivariate normal model summary statistics with `n` observations, mean `m` and
sample covariance `S`. Only saves the summary statistics.
"""
struct MvNormalData{TW <: Real, Tm <: AbstractVector, TS <: AbstractMatrix}
    "sum of the weights (alternatively, total number of observations)"
    W::TW
    "(weighted) sample mean"
    m::Tm
    "(weighted) sample covariance matrix"
    S::TS
    function MvNormalData(W::TW, m::Tm, S::TS) where {TW <: Real,
                                                      Tm <: AbstractVector,
                                                      TS <: AbstractMatrix}
        @argcheck is_conformable_square(m, S)
        new{TW, Tm, TS}(W, m, S)
    end
end

size(ss::MvNormalData) = ss.W, length(ss.m)

function show(io::IO, ss::MvNormalData)
    W, k = size(ss)
    print(io, "Summary statistics for multivariate normal, $(W) × $(k) samples")
end

"""
    $SIGNATURES

Multivariate normal summary statistics from observations (each row of `X` is an
observation).
"""
function MvNormalData(X::AbstractMatrix)
    m, S = mean_and_cov(X, 1; corrected = false)
    MvNormalData(size(X, 1), vec(m), S)
end

"""
    $SIGNATURES

Multivariate normal summary statistics from observations (each row of `X` is an
observation), with weights.
"""
function MvNormalData(X::AbstractMatrix, wv::AbstractWeights)
    m, S = mean_and_cov(X, wv, 1; corrected = false)
    MvNormalData(sum(wv), vec(m), S)
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

MLE(::MvNormalModel, ss::MvNormalData) = MvNormalParams(ss.m, ss.S)

function loglikelihood(::MvNormalModel, ss::MvNormalData, params::MvNormalParams)
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
Model data with a scalar- or vector-valued ordinary least squares regression.
See [`OLSData`](@ref) for wrapping data.
"""
struct OLSModel end

"""
    OLSData(Y, X)

Ordinary least squares with dependent variable `Y` and design matrix `X`.

Either

1. `Y` is an ``n×m`` matrix, then ``Y = X B + E`` where `X` is a ``n×k`` matrix,
`B` is a ``k×m`` parameter matrix, and ``Eᵢ ∼ N(0, Σ)`` is IID with ``m×m``
variance matrix ``Σ`` (multivariate linear regression), or

2. `Y` is a length `n` vector, then ``Y = X β + ϵ``, where `X` is a ``n×k``
matrix, `β` is a parameter vector of `k` elements, and `ϵᵢ ∼ N(0, σ)` where `σ`
is the variance of the normal error.

See [`OLSParams`](@ref) for the parameters.
"""
struct OLSData{TY <: Union{AbstractMatrix, AbstractVector},
               TX <: AbstractMatrix}
    Y::TY
    X::TX
    function OLSData(Y::TY, X::TX) where {TY, TX}
        @argcheck size(Y, 1) == size(X, 1)
        new{TY, TX}(Y, X)
    end
end

function show(io::IO, data::OLSData{<:AbstractVector})
    n, k = size(data.X)
    print(io, "OLS regression data, $(n) scalar observations, $(k) covariates")
end

function show(io::IO, data::OLSData{<:AbstractMatrix})
    n, k = size(data.X)
    m = size(data.Y, 2)
    print(io, "OLS regression data, $(n) vector observations of length $(m), $(k) covariates")
end

"""
    OLSParams(B, Σ)

Maximum likelihood estimated parameters for an OLS regression. See
[`OLSData`](@ref).
"""
struct OLSParams{TB <: Union{AbstractMatrix, AbstractVector},
                  TΣ <: Union{<:Real, AbstractMatrix}}
    B::TB
    Σ::TΣ
    function OLSParams(B::TB, Σ::TΣ) where {TB <: AbstractVector, TΣ <: Real}
        new{TB, TΣ}(B, Σ)
    end
    function OLSParams(B::TB, Σ::TΣ) where {TB <: AbstractMatrix,
                                             TΣ <: AbstractMatrix}
        m = size(B, 2)
        @argcheck size(Σ) == (m, m)
        new{TB, TΣ}(B, Σ)
    end
end

function show(io::IO, params::OLSParams{<:AbstractVector})
    k = length(params.B)
    print(io, "OLS regression parameters, $(k) covariates")
end

function show(io::IO, params::OLSParams{<:AbstractMatrix})
    k, m = size(params.B)
    print(io, "OLS regression parameters, vector observations of length $(m), $(k) covariates")
end

function MLE(::OLSModel, data::OLSData)
    @unpack Y, X = data
    B = qrfact(X, Val{true}) \ Y
    E = Y - X*B
    Σ = E'*E / size(X, 1)
    OLSParams(B, Σ)
end

function loglikelihood(::OLSModel, data::OLSData{<: AbstractVector},
                       params::OLSParams{<: AbstractVector})
    @unpack Y, X = data
    @unpack B, Σ = params
    n = length(Y)
    @argcheck size(X, 1) == n
    @argcheck size(X, 2) == length(B)
    E = Y - X * B
    -0.5 * (n*(log(2*π) + log(Σ)) + sum(abs2, E)/Σ)
end

"""
    $SIGNATURES

Sum of the log pdf for observations ``xᵢ ∼ MultivariateNormal(0, Σ)``, where
``xᵢ`` are stored in the rows of `X`.

Handles `ForwardDiff.Dual` types and singular covariance matrices. Helper
function, not part of the API.
"""
function _logpdf_normal(X::AbstractMatrix{TX},
                        Σ::AbstractMatrix{TΣ}) where {TX, TΣ}
    n, m = size(X)
    try
        U = chol(Σ)
        A = X / U
        -0.5 * (n*(m*log(2*π) + 2*logdet(U)) + sum(abs2, A))
    catch
        convert(promote_type(TX, TΣ), -Inf)
    end
end

function loglikelihood(::OLSModel, data::OLSData{<: AbstractMatrix},
                       params::OLSParams{<: AbstractMatrix})
    @unpack Y, X = data
    @unpack B, Σ = params
    n, m = size(Y)
    @argcheck size(B, 1) == size(X, 2) # k
    @argcheck size(B, 2) == m
    _logpdf_normal(Y - X * B, Σ)
end

"""
    $SIGNATURES

Add an intercept to a matrix or vector of covariates.
"""
add_intercept(X::AbstractMatrix{T}) where T = hcat(ones(T, size(X, 1)), X)

add_intercept(x::AbstractVector{T}) where T = hcat(ones(T, length(x)), x)

add_intercept(xs...) = add_intercept(hcat(xs...))


# modeling API

"""
Abstract type for indirect likelihood problems.
"""
abstract type AbstractIndirectLikelihoodProblem end

"""
    IndirectLikelihoodProblem(structural_model, auxiliary_model, observed_data)

A simple wrapper for an indirect likelihood problem.

## Terminology

The *log_prior* is a callable that returns the log prior for a set of structural
parameters ``θ``.

A *structural model*, common random numbers ``ϵ``, and a set of parameters ``θ``
are sufficient to generate *simulated data*, by implementing a method for
[`simulate_data`](@ref). When applicable, independent variables necessary for
simulation should be included in the *structural model*.

*Common random numbers* are a set of random numbers that can be re-used for for
simulation with different parameter values. [`random_crn`] should yield an
initial value (usually an `Array` or a collection of similar structures), while
[`random_crn!`](@ref) should be able to update this in place. The latter is used
primarily for indirect loglikelihood calculations. When the model does not admit
common random numbers, use `nothing`.

An *auxiliary model* is a simpler model for which we can calculate the
likelihood. Given simulated data `x`, a maximum likelihood parameter estimate
``ϕ`` is obtained with `MLE(auxiliary_model, x)`. The log likelihood of
*observed data* `y` is calculated with `loglikelihood(auxiliary_model, y, ϕ)`.

A *problem* is defined by its structural and auxiliary model, and the observed
data. Objects of this type are *callable*: when called with the parameters
(which may be a tuple, or any kind of problem-dependent structure accepted by
[`simulate_data`](@ref)), it should return the indirect log likelihood. The
default callable method does this.

The user should implement [`simulate_data`](@ref), [`MLE`](@ref), and
[`loglikelihood`](@ref), with the signatures above.
"""
struct IndirectLikelihoodProblem{P, S, E, A, D} <: AbstractIndirectLikelihoodProblem
    log_prior::P
    structural_model::S
    common_random_numbers::E
    auxiliary_model::A
    observed_data::D
end

"""
    $SIGNATURES

Simulate data from the `structural_model` with the given parameters `θ`. Common
random numbers `ϵ` are used when provided, otherwise generated using
[`random_crv`](@ref).

Methods should be defined by the user for each `structural_model` type. The
function should return simulated `data` in the format that can be used by
`MLE(auxiliary_model, data)`.

For infeasible/meaningless parameters, `nothing` should be returned.
"""
@no_method_info simulate_data(structural_model, θ, ϵ)

"""
    $SIGNATURES

Return a container of common random numbers that can be reused by
[`simulate_data`](@ref) with different parameters.

The returned value does not need to be a mutable type, but its contents need to
be modifiable. A method for [`random_crn!`](@ref) should be defined which can
update them.

When the model structure does not allow common random numbers, return
`nothing`, which already has a fallback method for [`random_crn!`](@ref).
"""
@no_method_info random_crn(structural_model)

"""
    $SIGNATURES

Update the common random numbers for the model.
"""
@no_method_info random_crn!(structural_model, ϵ)

random_crn!(::Any, ::Void) = nothing

random_crn!(problem::IndirectLikelihoodProblem) =
    random_crn!(problem.structural_model, problem.common_random_numbers)

"""
    $SIGNATURES

Maximum likelihood estimate for the auxiliary model with `data`.

Methods should be defined by the user for each `auxiliary_model` type. A
fallback method is defined for `nothing` as data (infeasible parameters). See
also [`simulate_data`](@ref).
"""
@no_method_info MLE(auxiliary_model, data)

MLE(::Any, ::Void) = nothing

"""
    $SIGNATURES

Return an estimate of the auxiliary parameters `ϕ` as a function of the
structural parameters `θ`.

This is also known the *mapping* or *binding function* in the indirect inference
literature.
"""
function indirect_estimate(problem::IndirectLikelihoodProblem, θ)
    @unpack structural_model, common_random_numbers, auxiliary_model = problem
    x = simulate_data(structural_model, θ, common_random_numbers)
    MLE(auxiliary_model, x)
end


function (problem::IndirectLikelihoodProblem)(θ)
    @unpack log_prior, auxiliary_model, observed_data = problem
    ϕ = indirect_estimate(problem, θ)
    if ϕ == nothing
        -Inf
    else
        loglikelihood(auxiliary_model, observed_data, ϕ) + log_prior(θ)
    end
end

"""
    $SIGNATURES

Initialize an [`IndirectLikelihoodProblem`](@ref) with simulated data, using
parameters `θ`.

Useful for debugging and exploration of identification with simulated data.
"""
function simulate_problem(log_prior, structural_model, auxiliary_model, θ;
                          ϵ = random_crn(structural_model))
    IndirectLikelihoodProblem(log_prior, structural_model, ϵ, auxiliary_model,
                              simulate_data(structural_model, θ, ϵ))
end


# local analysis

"""
    $SIGNATURES

Return the values of the argument as a vector, potentially (but not necessarily)
restricting to those elements that uniquely determine the argument.

For example, a symmetric matrix would be determined by the diagonal and either
half.
"""
function vec_parameters end

vec_parameters(x::Real) = [x]

vec_parameters(xs::Tuple) = vcat(map(vec_parameters, xs)...)

vec_parameters(xs...) = vec_parameters(xs)

vec_parameters(A::AbstractArray) = vec(A)

"""
    $SIGNATURES

Flattened elements from the upper triangle. Helper function for
[`vec_parameters`](@ref).
"""
function vec_upper(U::AbstractMatrix{T}) where T
    n = LinAlg.checksquare(U)
    l = n*(n+1) ÷ 2
    v = Vector{T}(l)
    k = 1
    @inbounds for i in 1:n
        for j in 1:i
            v[k] = U[j, i]
            k += 1
        end
    end
    v
end

"""
    $SIGNATURES

Flattened elements from the lower triangle. Helper function for
[`vec_parameters`](@ref).
"""
function vec_lower(L::AbstractMatrix{T}) where T
    n = LinAlg.checksquare(L)
    l = n*(n+1) ÷ 2
    v = Vector{T}(l)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            v[k] = L[j, i]
            k += 1
        end
    end
    v
end

vec_parameters(A::Union{Symmetric, UpperTriangular}) = vec_upper(A)

vec_parameters(A::LowerTriangular) = vec_lower(A)

vec_parameters(ϕ::OLSParams) = vec_parameters(ϕ.B, ϕ.Σ)

"""
    $SIGNATURES

Calculate the local Jacobian of the estimated auxiliary parameters ``ϕ`` at the
structural parameters ``θ=θ₀``.

`ω_to_θ` maps a vector of reals `ω` to the parameters `θ` in the format
acceptable to `structural_model`. It should support
[`ContinuousTransformations.transform`](@ref) and
[`ContinuousTransformations.inverse`](@ref). See, for example,
[`ContinuousTransformations.TupleTransformation`](@ref).

`vecϕ` is a function that is used to flatten the auxiliary parameters to a
vector. Defaults to [`vec_parameters`](@ref).
"""
function local_jacobian(problem::IndirectLikelihoodProblem, θ₀, ω_to_θ;
                        vecϕ = vec_parameters)
    @unpack structural_model, auxiliary_model = problem
    ω₀ = inverse(ω_to_θ, θ₀)
    jacobian(ω₀) do ω
        θ = transform(ω_to_θ, ω)
        ϕ = indirect_estimate(problem, θ)
        vecϕ(ϕ)
    end
end

end # module
