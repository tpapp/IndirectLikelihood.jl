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


# test utilities

@testset "fallback methods" begin
    output = @capture_err begin
        @test_throws MethodError MLE("something", "nonsensical")
    end
    @test contains(output, "You need to define `MLE` with this model type.")
end

const RNG = Base.Random.GLOBAL_RNG

firstline(lines) = split(lines, '\n')[1]


# test auxiliary model building blocks

@testset "MvNormalData summary statistics and MLE" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(N, K)
        mvn = MvNormalData(X)
        @test mvn.W == N
        @test size(mvn.m) == (K, )
        @test mvn.m ≈ vec(mean(X, 1))
        Xc = X .- mvn.m'
        @test mvn.S ≈ (Xc'*Xc)/N
        @test MLE(MvNormalModel(), mvn) == MvNormalParams(mvn.m, mvn.S)
        @test size(mvn) == (N, K)
        @test firstline(repr(mvn)) ==
            "Summary statistics for multivariate normal, $(N) × $(K) samples"
    end
end

@testset "MvNormalData summary statistics and MLE (weighted)" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(N, K)
        wv = Weights(abs.(randn(N)))
        W = sum(wv)
        mvn = MvNormalData(X, wv)
        @test mvn.W == W
        @test size(mvn.m) == (K, )
        @test mvn.m ≈ vec(mean(X, wv, 1))
        Xc = X .- mvn.m'
        @test mvn.S ≈ ((values(wv).*Xc)'*Xc)/W
        @test MLE(MvNormalModel(), mvn) == MvNormalParams(mvn.m, mvn.S)
        @test size(mvn) == (W, K)
        @test firstline(repr(mvn)) ==
            "Summary statistics for multivariate normal, $(sum(wv)) × $(K) samples"
    end
end

@testset "MvNormalData log likelihood" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(2*N, K)
        mvnX = MvNormalData(X)
        paramsX = MLE(MvNormalModel(), mvnX)
        Y = randn(N, K)
        mvnY = MvNormalData(Y)
        ℓ = sum(logpdf(MvNormal(paramsX.μ, paramsX.Σ), Y'))
        @test ℓ ≈ loglikelihood(MvNormalModel(), mvnY, paramsX)
        @test ℓ ≈ indirect_loglikelihood(MvNormalModel(), mvnY, mvnX)
    end
end

@testset "OLS scalar variate" begin
    for _ in 1:100
        k = rand(5:10)
        n = rand(90:100) + k
        X = randn(n, k)
        β = randn(k)
        Y = X * β + randn(n)
        data = OLSData(Y, X)
        @test repr(data) ==
            "OLS regression data, $(n) scalar observations, $(k) covariates"
        @test_throws ArgumentError OLSData(randn(n+1), X)
        params = MLE(OLSModel(), data)
        @test repr(params) == "OLS regression parameters, $(k) covariates"
        @test params.B ≈ (X \ Y)
        @test params.Σ ≈ sum(abs2, Y - X * (X \ Y)) / n
        for _ in 1:10
            β′ = randn(k)
            Σ′ = abs(randn()) + 1
            @test loglikelihood(OLSModel(), data, OLSParams(β′, Σ′)) ≈
                sum(logpdf.(Normal(0, √Σ′), Y - X * β′))
        end
    end
end

@testset "OLS matrix variate" begin
    m = rand(3:6)
    k = rand(5:10)
    n = rand(90:100) + k
    A = randn(m, m)
    X = randn(n, k)
    B = randn(k, m)
    Y = X * B + randn(n, m) * A
    data = OLSData(Y, X)
    @test repr(data) ==
        "OLS regression data, $(n) vector observations of length $(m), $(k) covariates"
    @test_throws ArgumentError OLSData(randn(n+1, ), X)
    params = MLE(OLSModel(), data)
    @test repr(params) ==
        "OLS regression parameters, vector observations of length $(m), $(k) covariates"
    @test params.B ≈ (X \ Y)
    E = Y - X * (X \ Y)
    @test params.Σ ≈ E'*E / n
    for _ in 1:10
        B′ = randn(k, m)
        A′ = randn(m, m)
        Σ′ = A′' * A′           # ensures PSD
        @test loglikelihood(OLSModel(), data, OLSParams(B′, Σ′)) ≈
            StatsBase.loglikelihood(MvNormal(zeros(m), Σ′), (Y - X * B′)')
    end
    # corner case: zero estimated variance matrix
    params2 = MLE(OLSModel(), data)
    params2.Σ .= 0
    @test loglikelihood(OLSModel(), data, params2) == -Inf
end

@testset "adding intercept" begin
    N = 11
    x = randn(N)
    X = randn(N, 3)
    x2 = randn(N)
    @test add_intercept(x) == hcat(ones(N), x)
    @test add_intercept(X) == hcat(ones(N), X)
    @test add_intercept(x, x2) == hcat(ones(N), x, x2)
    @test_throws DimensionMismatch add_intercept(x, ones(N+1))
end

@testset "vectorizing parameters" begin
    A = Matrix(reshape(1:9, 3, :))
    @test vec_parameters(UpperTriangular(A)) == [1, 4, 5, 7, 8, 9]
    @test vec_parameters(LowerTriangular(A)) == [1, 2, 3, 5, 6, 9]
    @test vec_parameters(Symmetric(A)) == [1, 4, 5, 7, 8, 9] # may change
    @test vec_parameters(A) == Vector(1:9)
    @test vec_parameters(1.0) == [1.0]
    @test vec_parameters((1.0, [2, 3])) == [1.0, 2.0, 3.0]
end

@testset "vectorizing parameters (OLS)" begin
    b = randn(rand(3:10))
    σ² = abs(randn())
    @test vec_parameters(OLSParams(b, σ²)) == vcat(b, σ²)
    B = randn(10, 3)
    Σ = randn(3, 3) |> x->x*x'
    @test vec_parameters(OLSParams(B, Σ)) == vcat(vec_parameters(B),
                                                  vec_parameters(Σ))
end


# problem API

"""
Observations from `Normal(μ, 1)` with an unknown `μ`, estimated using a
multivatiate normal auxiliary model (so in this case, the likelihood is exact
and the summary statistics lose no information).
"""
struct NormalMeanModel
    "number of observations"
    N::Int
end

common_random(rng::AbstractRNG, model::NormalMeanModel) = randn(rng, model.N)

common_random!(rng::AbstractRNG, model::NormalMeanModel, ϵ) = randn!(rng, ϵ)

function simulate_data(model::NormalMeanModel, θ, ϵ)
    μ = θ[1]
    MvNormalData(reshape(μ .+ ϵ, :, 1))
end

simulate_data(::NormalMeanModel, ::String, ::Any) = nothing

@testset "indirect likelihood toy problem" begin
    μ₀ = 2.0
    p = simulate_problem(x -> zero(x), NormalMeanModel(100), MvNormalModel(), [μ₀])
    # find the optimum using the same common random numbers
    f(x) = (-p([x]))[1]
    o = optimize(f, 0.0, 5.0)
    μ₁ = Optim.minimizer(o)
    @test μ₁ ≈ μ₀ atol = 1e-4
    # test common random numbers updating
    @test mean([(p = common_random!(RNG, p); mean(p.ϵ))
                for _ in 1:1000]) ≈ 0 atol = 0.01
    # invalid parameters: when data is nothing, indirect log likelihood is -Inf
    @test p("a fish") == -Inf
    # local jacobian
    parameter_transformation = TransformationTuple((IDENTITY, ))
    @test local_jacobian(p, (2.0, ), parameter_transformation) == [1.0 0.0]'
end
