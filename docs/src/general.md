# General interface

## Terminology

The *log_prior* is a callable that returns the log prior for a set of structural
parameters ``θ``.

An *auxiliary model* is a simpler model for which we can calculate the
likelihood. Given simulated data `x`, a maximum likelihood parameter estimate
``ϕ`` is obtained with `MLE(auxiliary_model, x)`. The log likelihood of
*observed data* `y` is calculated with `loglikelihood(auxiliary_model, y, ϕ)`.

A *problem* is defined by its structural and auxiliary model, and the observed
data. Objects of this type are *callable*: when called with the parameters
(which may be a tuple, or any kind of problem-dependent structure accepted by
[`simulate_data`](@ref)), it should return the indirect log likelihood. The
default callable method does this.


## Structural models

A **structural model** can be used to generate data from parameters ``θ`` and random numbers ``ϵ`` using [`simulate_data`](@ref). The mapping is not necessarily deterministic even given ``ϵ``, but the latter can be used for *common random numbers* for certain models to make the mapping continuous or reduce variance in simulations.

When applicable, independent variables (eg covariates) necessary for simulation should be included in the *structural model*. It is recommended that a type is defined for each problem, along with methods for [`simulate_data`](@ref).

```@docs
simulate_data
```

## Common random numbers

*Common random numbers* are a set of random numbers that can be re-used for for
simulation with different parameter values.

[`generate_crn`](@ref) should yield an initial value (usually an `Array` or a collection of similar structures). This is used when initializing indirect likelihood problems. `update_crn!` will be called to update them as necessary. The setup admits both mutable and immutable types.

```@docs
generate_crn
update_crn!
```

```@docs
MLE
indirect_loglikelihood
loglikelihood
IndirectLikelihoodProblem
simulate_problem
local_jacobian
vec_parameters
```
