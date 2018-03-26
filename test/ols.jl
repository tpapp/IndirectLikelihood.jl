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

@testset "vectorizing parameters (OLS)" begin
    b = randn(rand(3:10))
    σ² = abs(randn())
    @test vec_parameters(OLSParams(b, σ²)) == vcat(b, σ²)
    B = randn(10, 3)
    Σ = randn(3, 3) |> x->x*x'
    @test vec_parameters(OLSParams(B, Σ)) == vcat(vec_parameters(B),
                                                  vec_parameters(Σ))
end
