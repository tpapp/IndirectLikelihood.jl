export OLSModel, OLSData, OLSParams, add_intercept

"""
Model data with a scalar- or vector-valued ordinary least squares regression.
See [`OLSData`](@ref) for wrapping data.
"""
struct OLSModel end

"""
    OLSData(Y, X)

Ordinary least squares with dependent variable `Y` and design matrix `X`.

Either

1. ``Y`` is an ``n×m`` matrix, then ``Y = X B + E`` where ``X`` is a ``n×k``
    matrix, ``B`` is a ``k×m`` parameter matrix, and ``Eᵢ ∼ N(0, Σ)`` is IID
    with ``m×m`` variance matrix ``Σ`` (multivariate linear regression), or

2. ``Y`` is a length ``n`` vector, then ``Y = X β + ϵ``, where ``X`` is a
    ``n×k`` matrix, ``β`` is a parameter vector of ``k`` elements, and
    ``ϵᵢ∼N(0,σ)`` where ``σ`` is the variance of the normal error.

See also [`add_intercept`](@ref).
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
        # NOTE: technically, should check for X=0 in this case, but that almost
        # never happens and would need to throw an error, -Inf rejects without a
        # hassle too
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

vec_parameters(ϕ::OLSParams) = vec_parameters((ϕ.B, ϕ.Σ))
