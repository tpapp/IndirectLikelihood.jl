export MvNormalModel, MvNormalData, MvNormalParams

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

struct MvNormalData{TW <: Real, Tm <: AbstractVector, TS <: AbstractMatrix}
    "sum of the weights (alternatively, total number of observations)"
    W::TW
    "(weighted) sample mean"
    m::Tm
    "(weighted) sample covariance matrix"
    S::TS
    @doc """
        MvNormalData(n, m, S)

    Multivariate normal model summary statistics with `n` observations, mean `m`
    and sample covariance `S`. Only saves the summary statistics.

    !!! usage

        Use `MvNormalData(X, [wv])` to construct from data.
    """ ->                      # FIXME workaround, remove -> in v0.7
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
    println(io, "Summary statistics for multivariate normal, $(W) × $(k) samples")
    println(io, "sample mean: $(ss.m)")
    println(io, "sample covariance: $(ss.S)")
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

struct MvNormalParams{Tμ, TΣ}
    "mean"
    μ::Tμ
    "variance"
    Σ::TΣ
    @doc """
        MvNormalParams(μ, Σ)

    Parameters for the multivariate normal model ``x ∼ MvNormal(μ, Σ)``.

    !!! usage

        Construct using [`MLE(::MvNormalModel, ::MvNormalData)`](@ref).
    """ ->                      # FIXME workaround, remove -> in v0.7
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

vec_parameters(ϕ::MvNormalParams) = vec_parameters((ϕ.μ, ϕ.Σ))
