# [General interface](@id general_interface)

## Overview and concepts

The general interface supports the following setup for likelihood-based indirect inference, using a **structural model** ``S`` and an **auxiliary model** ``A``, with given **data** ``y``.

For each set of **parameters** ``θ``, a **structural model** ``S`` is used to generate **simulated data** ``x``, ie
```math
x_S(θ, ϵ)
```
where ``ϵ`` is a set of [common random numbers](@ref common_random_numbers) that can be kept constant for various ``θ``s. Nevertheless, the above is not necessarily a deterministic relationship, as additional randomness can be used.

An **auxiliary model** ``A`` with parameters ``ϕ`` can be estimated using generated data ``x`` with maximum likelihood, ie
```math
ϕ_A(x) = \arg\max_ϕ p_A(x ∣ ϕ)
```
is the maximum likelihood estimate.

The likelihood of ``y`` at parameters ``θ`` is obtained as
```math
p_A(y ∣ ϕ_A(x_S(θ, ϵ))
```

This is multiplied by a prior in ``θ``, specified in logs.

The **user should define**

1. **types** for the *structural model* and the *auxiliary model*,

2. **methods** for the functions below to dispatch on these types.

| component                   | Julia method            |
|:----------------------------|:------------------------|
| ``x_M``                     | [`simulate_data`](@ref) |
| ``p_A``                     | [`loglikelihood`](@ref) |
| ``ϕ_A``                     | [`MLE`](@ref)           |
| draw ``ϵ``                  | [`common_random`](@ref)  |

The framework is explained in detail below.

## [Data](@id data)

Data can be of any type, since it is generated and used by user-defined functions. Arrays, tuples (optionally named) are recommended for simple models, as there no need to wrap them in a type, as the auxiliary model type is used for dispatch.

More complex data structures may benefit from being wrapped in a `struct`.

## Structural models

A structural model can be used to generate [data](@ref data) from parameters ``θ`` and random numbers ``ϵ`` using [`simulate_data`](@ref). The mapping is not necessarily deterministic even given ``ϵ``, but the latter can be used for [common random numbers](@ref common_random_numbers) for certain models, to make the mapping continuous, or reduce variance in simulations.

When applicable, independent variables (eg covariates) necessary for simulation should be included in the structural model object. It is recommended that a type is defined for each problem, along with methods for [`simulate_data`](@ref).

```@docs
simulate_data
```

## [Common random numbers](@id common_random_numbers)

Common random numbers are a set of random numbers that can be re-used for for
simulation with different parameter values.

[`common_random`](@ref) should yield a random value for the random variables (usually an `Array` or a collection of similar structures).

```@docs
common_random
```

For error structures which can be overwritten in place, the user can define [`common_random!`](@ref) as an optimization.

```@docs
common_random!
```

## Likelihood

These methods should be defined for stuctural model types, and accept data from [`simulate_data`](@ref).

```@docs
MLE
loglikelihood
```

## Problem framework

```@docs
IndirectLikelihoodProblem
simulate_problem
```

## Utilities

```@docs
local_jacobian
vec_parameters
indirect_loglikelihood
```
