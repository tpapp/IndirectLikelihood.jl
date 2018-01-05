using IndirectLikelihood

using IndirectLikelihood:
    # multivariate normal
    MvNormal_SS, MvNormal_Params,
    # OLS
    OLS_Data, OLS_Params, add_intercept,
    # imported for testing
    vec_parameters

using Base.Test

using StatsBase: Weights, loglikelihood
using Distributions: logpdf, Normal, MvNormal

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
            "OLS regression data, $(n) scalar observations, $(k) covariates"
        @test_throws ArgumentError OLS_Data(randn(n+1), X)
        params = MLE(data)
        @test repr(params) == "OLS regression parameters, $(k) covariates"
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
        "OLS regression data, $(n) vector observations of length $(m), $(k) covariates"
    @test_throws ArgumentError OLS_Data(randn(n+1, ), X)
    params = MLE(data)
    @test repr(params) ==
        "OLS regression parameters, vector observations of length $(m), $(k) covariates"
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
