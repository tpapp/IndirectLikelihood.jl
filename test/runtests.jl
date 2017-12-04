import IndirectLikelihood:
    MLE, loglikelihood, indirect_loglikelihood,
    MvNormal_SS, MvNormal_Params, OLS_Data, OLS_Params
using Base.Test
using StatsBase
using Distributions

@testset "MvNormal_SS summary statistics and MLE" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(N, K)
        mvn = MvNormal_SS(X)
        @test mvn.W == N
        @test size(mvn.m) == (K, )
        @test mvn.m ≈ vec(mean(X, 1))
        Xc = X .- mvn.m'
        @test mvn.S ≈ (Xc'*Xc)/N
        @test MLE(mvn) == MvNormal_Params(mvn.m, mvn.S)
        @test size(mvn) == (N, K)
        @test repr(mvn) ==
            "Summary statistics for multivariate normal, $(N) × $(K) samples"
    end
end

@testset "MvNormal_SS summary statistics and MLE (weighted)" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(N, K)
        wv = Weights(abs.(randn(N)))
        W = sum(wv)
        mvn = MvNormal_SS(X, wv)
        @test mvn.W == W
        @test size(mvn.m) == (K, )
        @test mvn.m ≈ vec(mean(X, wv, 1))
        Xc = X .- mvn.m'
        @test mvn.S ≈ ((values(wv).*Xc)'*Xc)/W
        @test MLE(mvn) == MvNormal_Params(mvn.m, mvn.S)
        @test size(mvn) == (W, K)
        @test repr(mvn) ==
            "Summary statistics for multivariate normal, $(sum(wv)) × $(K) samples"
    end
end

@testset "MvNormal_SS log likelihood" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(2*N, K)
        mvnX = MvNormal_SS(X)
        paramsX = MLE(mvnX)
        Y = randn(N, K)
        mvnY = MvNormal_SS(Y)
        ℓ = sum(logpdf(MvNormal(paramsX.μ, paramsX.Σ), Y'))
        @test ℓ ≈ loglikelihood(mvnY, paramsX)
        @test ℓ ≈ indirect_loglikelihood(mvnY, mvnX)
    end
end

@testset "OLS scalar variate" begin
    for _ in 1:100
        k = rand(5:10)
        n = rand(90:100) + k
        X = randn(n, k)
        β = randn(k)
        Y = X * β + randn(n)
        data = OLS_Data(Y, X)
        @test repr(data) ==
            "OLS regression, $(n) scalar observations, $(k) covariates"
        @test_throws ArgumentError OLS_Data(randn(n+1), X)
        params = MLE(data)
        @test params.B ≈ (X \ Y)
        @test params.Σ ≈ sum(abs2, Y - X * (X \ Y)) / n
        for _ in 1:10
            β′ = randn(k)
            Σ′ = abs(randn()) + 1
            @test loglikelihood(data, OLS_Params(β′, Σ′)) ≈
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
    data = OLS_Data(Y, X)
    @test repr(data) ==
        "OLS regression, $(n) vector observations of length $(m), $(k) covariates"
    @test_throws ArgumentError OLS_Data(randn(n+1, ), X)
    params = MLE(data)
    @test params.B ≈ (X \ Y)
    E = Y - X * (X \ Y)
    @test params.Σ ≈ E'*E / n
    for _ in 1:10
        B′ = randn(k, m)
        A′ = randn(m, m)
        Σ′ = A′' * A′           # ensures PSD
        @test loglikelihood(data, OLS_Params(B′, Σ′)) ≈
            loglikelihood(MvNormal(zeros(m), Σ′), (Y - X * B′)')
    end
end
