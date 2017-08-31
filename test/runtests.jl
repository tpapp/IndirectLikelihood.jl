using IndirectLikelihood
using Base.Test
using StatsBase
using Distributions

@testset "MvNormalSS constructor and ML" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        X = randn(N, K)
        mvn = MvNormalSS(X)
        @test mvn.n == N
        @test size(mvn.m) == (K, )
        @test mvn.m ≈ vec(mean(X, 1))
        Xc = X .- mvn.m'
        @test mvn.S ≈ (Xc'*Xc)/N
        @test ML(mvn) == (mvn.m, mvn.S)
        @test size(mvn) == (N, K)
        @test repr(mvn) == "Summary statistics for multivariate normal, $(N) × $(K) samples"
    end
end

@testset "MvNormalSS log likelihood" begin
    for _ in 1:100
        K = rand(5:10)
        N = K + rand(10:100)
        μ = randn(K)
        A = randn(K,K)
        Σ = A'*A
        Y = randn(N, K)
        @test sum(logpdf(MvNormal(μ, Σ), Y')) ≈ loglikelihood(MvNormalSS(Y), μ, Σ)
    end
end