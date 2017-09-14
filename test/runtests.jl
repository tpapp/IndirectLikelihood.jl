using IndirectLikelihood
using Base.Test
using StatsBase
using Distributions

@testset "MvNormalSS summary statistics and ML" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(N, K)
        mvn = summary_statistics(MvNormalSS, X)
        @test mvn.W == N
        @test size(mvn.m) == (K, )
        @test mvn.m ≈ vec(mean(X, 1))
        Xc = X .- mvn.m'
        @test mvn.S ≈ (Xc'*Xc)/N
        @test ML(mvn) == MvNormalParams(mvn.m, mvn.S)
        @test size(mvn) == (N, K)
        @test repr(mvn) ==
            "Summary statistics for multivariate normal, $(N) × $(K) samples"
    end
end

@testset "MvNormalSS summary statistics and ML (weighted)" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(N, K)
        wv = Weights(abs.(randn(N)))
        W = sum(wv)
        mvn = summary_statistics(MvNormalSS, X, wv)
        @test mvn.W == W
        @test size(mvn.m) == (K, )
        @test mvn.m ≈ vec(mean(X, wv, 1))
        Xc = X .- mvn.m'
        @test mvn.S ≈ ((values(wv).*Xc)'*Xc)/W
        @test ML(mvn) == MvNormalParams(mvn.m, mvn.S)
        @test size(mvn) == (W, K)
        @test repr(mvn) ==
            "Summary statistics for multivariate normal, $(sum(wv)) × $(K) samples"
    end
end

@testset "MvNormalSS log likelihood" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(2*N, K)
        mvnX = summary_statistics(MvNormalSS, X)
        paramsX = ML(mvnX)
        Y = randn(N, K)
        mvnY = summary_statistics(MvNormalSS, Y)
        ℓ = sum(logpdf(MvNormal(paramsX.μ, paramsX.Σ), Y'))
        @test ℓ ≈ loglikelihood(mvnY, paramsX)
        @test ℓ ≈ indirect_loglikelihood(mvnY, mvnX)
    end
end
