export
    # general interface
    MLE, indirect_loglikelihood, loglikelihood,
    # modeling API
    IndirectLikelihoodProblem, simulate_problem, loglikelihood, local_jacobian,
    simulate_data, random_crn, random_crn!, vec_parameters

"""
    $SIGNATURES

Return the maximum likelihood of the parameters for `data` in `model`. See also
the 3-parameter version of [`loglikelihood`](@ref) in this package.
"""
@no_method_info MLE(model, data)

"""
    $SIGNATURES

Log likelihood of `data` under `model` with `parameters`.
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

"""
Abstract type for indirect likelihood problems.
"""
abstract type AbstractIndirectLikelihoodProblem end

"""
    IndirectLikelihoodProblem(log_prior, structural_model, auxiliary_model, observed_data)

A simple wrapper for an indirect likelihood problem.

## Terminology

The *log_prior* is a callable that returns the log prior for a set of structural
parameters ``θ``.

A *structural model*, common random numbers ``ϵ``, and a set of parameters ``θ``
are sufficient to generate *simulated data*, by implementing a method for
[`simulate_data`](@ref). When applicable, independent variables necessary for
simulation should be included in the *structural model*.

*Common random numbers* are a set of random numbers that can be re-used for for
simulation with different parameter values. [`random_crn`](@ref) should yield an
initial value (usually an `Array` or a collection of similar structures), while
[`random_crn!`](@ref) should be able to update this in place. The latter is used
primarily for indirect loglikelihood calculations. When the model does not admit
common random numbers, use `nothing`.

An *auxiliary model* is a simpler model for which we can calculate the
likelihood. Given simulated data `x`, a maximum likelihood parameter estimate
``ϕ`` is obtained with `MLE(auxiliary_model, x)`. The log likelihood of
*observed data* `y` is calculated with `loglikelihood(auxiliary_model, y, ϕ)`.

A *problem* is defined by its structural and auxiliary model, and the observed
data. Objects of this type are *callable*: when called with the parameters
(which may be a tuple, or any kind of problem-dependent structure accepted by
[`simulate_data`](@ref)), it should return the indirect log likelihood. The
default callable method does this.

The user should implement [`simulate_data`](@ref), [`MLE`](@ref), and
[`loglikelihood`](@ref), with the signatures above.
"""
struct IndirectLikelihoodProblem{P, S, E, A, D} <: AbstractIndirectLikelihoodProblem
    log_prior::P
    structural_model::S
    common_random_numbers::E
    auxiliary_model::A
    observed_data::D
end

"""
    $SIGNATURES

Simulate data from the `structural_model` with the given parameters `θ`. Common
random numbers `ϵ` are used when provided, otherwise generated using
[`random_crv`](@ref).

Three-argument methods should be defined by the user for each `structural_model`
type. The function should return simulated `data` in the format that can be used
by `MLE(auxiliary_model, data)`. For infeasible/meaningless parameters,
`nothing` should be returned.
"""
@no_method_info simulate_data(structural_model, θ, ϵ)

simulate_data(structural_model, θ) =
    simulate_data(structural_model, θ, random_crn(structural_model))

"""
    $SIGNATURES

Return a container of common random numbers that can be reused by
[`simulate_data`](@ref) with different parameters.

The returned value does not need to be a mutable type, but its contents need to
be modifiable. A method for [`random_crn!`](@ref) should be defined which can
update them.

When the model structure does not allow common random numbers, return
`nothing`, which already has a fallback method for [`random_crn!`](@ref).
"""
@no_method_info random_crn(structural_model)

"""
    $SIGNATURES

Update the common random numbers for the model.
"""
@no_method_info random_crn!(structural_model, ϵ)

random_crn!(::Any, ::Void) = nothing

random_crn!(problem::IndirectLikelihoodProblem) =
    random_crn!(problem.structural_model, problem.common_random_numbers)

"""
    $SIGNATURES

Maximum likelihood estimate for the auxiliary model with `data`.

Methods should be defined by the user for each `auxiliary_model` type. A
fallback method is defined for `nothing` as data (infeasible parameters). See
also [`simulate_data`](@ref).
"""
@no_method_info MLE(auxiliary_model, data)

MLE(::Any, ::Void) = nothing

"""
    $SIGNATURES

Return an estimate of the auxiliary parameters `ϕ` as a function of the
structural parameters `θ`.

This is also known the *mapping* or *binding function* in the indirect inference
literature.
"""
function indirect_estimate(problem::IndirectLikelihoodProblem, θ)
    @unpack structural_model, common_random_numbers, auxiliary_model = problem
    x = simulate_data(structural_model, θ, common_random_numbers)
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
                          ϵ = random_crn(structural_model))
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
[`ContinuousTransformations.TupleTransformation`](@ref).

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
