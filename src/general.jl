export
    # model and common random numbers
    simulate_data, generate_crn, update_crn!,
    # likelihood
    MLE, indirect_loglikelihood, loglikelihood,
    # modeling API
    IndirectLikelihoodProblem, simulate_problem, loglikelihood, local_jacobian,
    simulate_data, generate_crn, update_crn, vec_parameters


# structural model and common random numbers

"""
    $SIGNATURES

Simulate data from the `structural_model` with the given parameters `θ`. Common
random numbers `ϵ` are used when provided, otherwise generated using
[`generate_crn`](@ref).

Three-argument methods should be defined by the user for each `structural_model`
type. The function should return simulated `data` in the format that can be used
by `MLE(auxiliary_model, data)`. For infeasible/meaningless parameters,
`nothing` should be returned.
"""
@no_method_info simulate_data(structural_model, θ, ϵ)

simulate_data(structural_model, θ) =
    simulate_data(structural_model, θ, generate_crn(structural_model))

"""
    $SIGNATURES

Return a container of common random numbers that can be reused by
[`simulate_data`](@ref) with different parameters. See also
[`update_crn!`](@ref).

When the model structure does not allow common random numbers, return `nothing`,
which already has a fallback method for [`update_crn!`](@ref).

The first argument is the random number generator, which defaults to [`RNG`](@ref).
"""
@no_method_info generate_crn(rng::AbstractRNG, structural_model)

"""
    $SIGNATURES
"""
generate_crn(structural_model) = generate_crn(RNG, structural_model)

"""
    $SIGNATURES

Update the common random numbers for the model. The semantics is as follows:

1. it *may*, but does not need to, change the contents of its second argument,

2. the new common random numbers should be returned regardless.

Two common usage patterns are

1. having a mutable `ϵ`, updating that in place, returning `ϵ`,

2. generating new `ϵ`, returning that.
"""
@no_method_info update_crn!(rng::AbstractRNG, structural_model, ϵ)

update_crn!(rng::AbstractRNG, ::Any, ::Void) = nothing

update_crn!(structural_model, ϵ) = update_crn!(RNG, structural_model, ϵ)


# problem interface

"""
Abstract type for indirect likelihood problems.
"""
abstract type AbstractIndirectLikelihoodProblem end

"""
    IndirectLikelihoodProblem(log_prior, structural_model, auxiliary_model, observed_data)

A simple wrapper for an indirect likelihood problem.


The user should implement [`simulate_data`](@ref), [`MLE`](@ref), and
[`loglikelihood`](@ref), with the signatures above.
"""
struct IndirectLikelihoodProblem{P, S, E, A, D} <: AbstractIndirectLikelihoodProblem
    log_prior::P
    structural_model::S
    ϵ::E
    auxiliary_model::A
    observed_data::D
end

"""
    $SIGNATURES

Return a new problem with updated common random numbers.
"""
function update_crn!(problem::IndirectLikelihoodProblem)
    @unpack log_prior, structural_model, ϵ, auxiliary_model, observed_data = problem
    ϵ′ = update_crn!(structural_model, ϵ)
    IndirectLikelihoodProblem(log_prior, structural_model, ϵ′, auxiliary_model,
                              observed_data)
end


# likelihood

"""
    $SIGNATURES

Maximum likelihood estimate of the parameters for `data` in `model`.

When `ϕ == MLE(model, data)`, `ϕ` should maximize `ϕ -> loglikelihood(model,
data, ϕ)`. See [`loglikelihood`](@ref).

Methods should be defined by the user for each `auxiliary_model` type. A
fallback method is defined for `nothing` as data (infeasible parameters). See
also [`simulate_data`](@ref).
"""
@no_method_info MLE(model, data)

MLE(::Any, ::Void) = nothing

"""
    $SIGNATURES

Log likelihood of `data` under `model` with `parameters`. See [`MLE`](@ref).
"""
@no_method_info loglikelihood(model, data, parameters)

"""
    indirect_loglikelihood(model, y, x)

1. estimate `model` from summary statistics `x` using maximum likelihood,

2. return the likelihood of summary statistics `y` under the estimated parameters.

Useful for pseudo-likelihood indirect inference, where `y` would be the observed
and `x` the simulated data. See in particular

- Gallant, A. R., & McCulloch, R. E. (2009). On the determination of general
  scientific models with application to asset pricing. Journal of the American
  Statistical Association, 104(485), 117–131.

- Drovandi, C. C., Pettitt, A. N., Lee, A., & others, (2015). Bayesian indirect
  inference using a parametric auxiliary model. Statistical Science, 30(1),
  72–95.
"""
indirect_loglikelihood(model, y, x) = loglikelihood(model, y, MLE(model, x))


# problem framework

"""
    $SIGNATURES

Return an estimate of the auxiliary parameters `ϕ` as a function of the
structural parameters `θ`.

This is also known the *mapping* or *binding function* in the indirect inference
literature.
"""
function indirect_estimate(problem::IndirectLikelihoodProblem, θ)
    @unpack structural_model, ϵ, auxiliary_model = problem
    x = simulate_data(structural_model, θ, ϵ)
    MLE(auxiliary_model, x)
end


function (problem::IndirectLikelihoodProblem)(θ)
    @unpack log_prior, auxiliary_model, observed_data = problem
    ϕ = indirect_estimate(problem, θ)
    if ϕ == nothing
        -Inf
    else
        loglikelihood(auxiliary_model, observed_data, ϕ) + log_prior(θ)
    end
end

"""
    $SIGNATURES

Initialize an [`IndirectLikelihoodProblem`](@ref) with simulated data, using
parameters `θ`.

Useful for debugging and exploration of identification with simulated data.
"""
function simulate_problem(log_prior, structural_model, auxiliary_model, θ;
                          ϵ = generate_crn(structural_model))
    IndirectLikelihoodProblem(log_prior, structural_model, ϵ, auxiliary_model,
                              simulate_data(structural_model, θ, ϵ))
end


# local analysis

"""
    $SIGNATURES

Return the values of the argument as a vector, potentially (but not necessarily)
restricting to those elements that uniquely determine the argument.

For example, a symmetric matrix would be determined by the diagonal and either
half.
"""
function vec_parameters end

vec_parameters(x::Real) = [x]

vec_parameters(xs::Tuple) = vcat(map(vec_parameters, xs)...)

vec_parameters(A::AbstractArray) = vec(A)

"""
    $SIGNATURES

Flattened elements from the upper triangle. Helper function for
[`vec_parameters`](@ref).
"""
function vec_upper(U::AbstractMatrix{T}) where T
    n = LinAlg.checksquare(U)
    l = n*(n+1) ÷ 2
    v = Vector{T}(l)
    k = 1
    @inbounds for i in 1:n
        for j in 1:i
            v[k] = U[j, i]
            k += 1
        end
    end
    v
end

"""
    $SIGNATURES

Flattened elements from the lower triangle. Helper function for
[`vec_parameters`](@ref).
"""
function vec_lower(L::AbstractMatrix{T}) where T
    n = LinAlg.checksquare(L)
    l = n*(n+1) ÷ 2
    v = Vector{T}(l)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            v[k] = L[j, i]
            k += 1
        end
    end
    v
end

vec_parameters(A::Union{Symmetric, UpperTriangular}) = vec_upper(A)

vec_parameters(A::LowerTriangular) = vec_lower(A)

"""
    $SIGNATURES

Calculate the local Jacobian of the estimated auxiliary parameters ``ϕ`` at the
structural parameters ``θ=θ₀``.

`ω_to_θ` maps a vector of reals `ω` to the parameters `θ` in the format
acceptable to `structural_model`. It should support
[`ContinuousTransformations.transform`](@ref) and
[`ContinuousTransformations.inverse`](@ref). See, for example,
[`ContinuousTransformations.TransformationTuple`](@ref).

`vecϕ` is a function that is used to flatten the auxiliary parameters to a
vector. Defaults to [`vec_parameters`](@ref).
"""
function local_jacobian(problem::IndirectLikelihoodProblem, θ₀, ω_to_θ;
                        vecϕ = vec_parameters)
    @unpack structural_model, auxiliary_model = problem
    ω₀ = inverse(ω_to_θ, θ₀)
    jacobian(ω₀) do ω
        θ = transform(ω_to_θ, ω)
        ϕ = indirect_estimate(problem, θ)
        vecϕ(ϕ)
    end
end
