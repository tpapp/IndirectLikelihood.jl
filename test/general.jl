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

function simulate_data(rng::AbstractRNG, model::NormalMeanModel, θ, ϵ)
    μ = θ[1]
    MvNormalData(reshape(μ .+ ϵ, :, 1))
end

simulate_data(rng::AbstractRNG, ::NormalMeanModel, ::String, ::Any) = nothing

MLE(::NormalMeanModel, data) = MLE(MvNormalModel(), data)

loglikelihood(::NormalMeanModel, data, ϕ) = loglikelihood(MvNormalModel(), data, ϕ)

@testset "indirect likelihood toy problem" begin
    μ₀ = 2.0
    p = simulate_problem(NormalMeanModel(100), zero ∘ first, [μ₀])
    # find the optimum using the same common random numbers
    f(x) = (-indirect_logposterior(p, [x]))[1]
    o = optimize(f, 0.0, 5.0)
    μ₁ = Optim.minimizer(o)
    @test μ₁ ≈ μ₀ atol = 1e-4
    # test common random numbers updating
    @test mean([(p = common_random!(p); mean(p.ϵ))
                for _ in 1:1000]) ≈ 0 atol = 0.01
    # invalid parameters: when data is nothing, indirect log likelihood is -Inf
    @test indirect_logposterior(p, "a fish") == -Inf
    # local jacobian
    parameter_transformation = TransformationTuple((IDENTITY, ))
    @test local_jacobian(p, (2.0, ), parameter_transformation) == [1.0 0.0]'
    # buildup of likelihood
    p2 = IndirectLikelihoodProblem(NormalMeanModel(100), zero ∘ first, p.data)
    θ = [μ₀]
    y = simulate_data(p2.rng, p2.model, θ, p2.ϵ)
    ϕ = MLE(p2.model, y)
    ℓ = loglikelihood(p2.model, p2.data, ϕ)
    @test indirect_logposterior(p2, θ) == ℓ
    @test indirect_logposterior(p2, θ) == indirect_logposterior(p2)(θ)
end

struct ToyModel end

common_random(rng::AbstractRNG, ::ToyModel) = 9.0

simulate_data(rng::AbstractRNG, ::ToyModel, ::Any, ::Any) = 42

@testset "simulation & CRN barebone" begin
    @test common_random(RNG, ToyModel()) == 9.0
    @test common_random!(RNG, ToyModel(), nothing) == 9.0 # fallback
    @test simulate_data(RNG, ToyModel(), :parameter) == 42
    @test simulate_data(ToyModel(), :parameter) == 42
end
