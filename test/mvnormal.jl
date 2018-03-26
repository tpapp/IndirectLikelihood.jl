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
