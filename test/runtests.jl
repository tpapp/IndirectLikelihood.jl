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
        X = randn(2*N, K)
        mvnX = summary_statistics(MvNormalSS, X)
        μ, Σ = ML(mvnX)
        Y = randn(N, K)
        mvnY = summary_statistics(MvNormalSS, Y)
        @test sum(logpdf(MvNormal(μ, Σ), Y')) ≈ loglikelihood(mvnY, μ, Σ)
        @test indirect_loglikelihood(mvnY, X) ≈ loglikelihood(mvnY, μ, Σ)
    end
end
