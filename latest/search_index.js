var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Introduction-1",
    "page": "Introduction",
    "title": "Introduction",
    "category": "section",
    "text": "This package implements a general framework for indirect inference using likelihood-based methods. It is composed of two parts:a very light general interface for organizing an indirect inference problem, mostly for avoiding repeated code,\nsome simple auxiliary models that can be used as building blocks (the interface is of course general enough to admit arbitrary models).This package assumes that you are familiar with the concept of indirect inference. If not, Smith (2008) is a good starting point, then see Drovandi et al (2015) for a survey on Bayesian methods, and Gallant & McCulloch (2009) for the particular method implemented by this package.In a nutshell, indirect inference is useful when you know how to simulate data from a given set of parameters for some structural model, but the problem is too complex to admit a direct likelihood-based representation. Instead, an auxiliary model is estimated on the actual and simulated data, and the distance between the two parameter estimates is minimized under some metric.The method of Gallant & McCulloch (2009) is particularly elegant because it fits into a likelihood-based framework, and induces a natural distance metric. Methods used in frequentist statistics/econometrics often rely on a weighting matrix instead, which needs to be estimated too, but it is not needed for this framework."
},

{
    "location": "index.html#References-1",
    "page": "Introduction",
    "title": "References",
    "category": "section",
    "text": "Drovandi, C. C., Pettitt, A. N., Lee, A., & others, (2015). Bayesian indirect inference using a parametric auxiliary model. Statistical Science, 30(1), 72–95. (preprint)\nGallant, A. R., & McCulloch, R. E. (2009). On the determination of general scientific models with application to asset pricing. Journal of the American Statistical Association, 104(485), 117–131. (preprint) (JSTOR)\nSmith, A. Indirect inference. The New Palgrave Dictionary of Economics, 2nd Edition (forthcoming) (2008)."
},

{
    "location": "general.html#",
    "page": "General interface",
    "title": "General interface",
    "category": "page",
    "text": ""
},

{
    "location": "general.html#general_interface-1",
    "page": "General interface",
    "title": "General interface",
    "category": "section",
    "text": ""
},

{
    "location": "general.html#Overview-and-concepts-1",
    "page": "General interface",
    "title": "Overview and concepts",
    "category": "section",
    "text": "The general interface supports the following setup for likelihood-based indirect inference, using a structural model S and an auxiliary model A, with given data y.For each set of parameters , a structural model S is used to generate simulated data x, iex_S( )where  is a set of common random numbers that can be kept constant for various s. Nevertheless, the above is not necessarily a deterministic relationship, as additional randomness can be used.An auxiliary model A with parameters  can be estimated using generated data x with maximum likelihood, ie_A(x) = argmax_ p_A(x  )is the maximum likelihood estimate.The likelihood of y at parameters  is obtained asp_A(y  _A(x_S( ))This is multiplied by a prior in , specified in logs.The user should definetypes for the structural model and the auxiliary model,\nmethods for the functions below to dispatch on these types.component Julia method\nx_M simulate_data\np_A loglikelihood\n_A MLE\ndraw  common_randomThe framework is explained in detail below."
},

{
    "location": "general.html#data-1",
    "page": "General interface",
    "title": "Data",
    "category": "section",
    "text": "Data can be of any type, since it is generated and used by user-defined functions. Arrays, tuples (optionally named) are recommended for simple models, as there no need to wrap them in a type, as the auxiliary model type is used for dispatch.More complex data structures may benefit from being wrapped in a struct."
},

{
    "location": "general.html#IndirectLikelihood.simulate_data",
    "page": "General interface",
    "title": "IndirectLikelihood.simulate_data",
    "category": "function",
    "text": "simulate_data(rng, structural_model, θ, ϵ)\n\n\nSimulate data from the structural_model\n\nusing random number generator rng,\nwith parameters θ,\ncommon random numbers ϵ.\n\nusage: Usage\nThe user should define a method for this function for each structural_model type with the signaturesimulate_data(rng::AbstractRNG, structural_model, θ, ϵ)This should return simulated data in the format that can be used by MLE and loglikelihood.For infeasible/meaningless parameters, nothing should be returned.\n\n\n\nsimulate_data(rng, structural_model, θ)\n\n\nSimulate data, generating ϵ using rng.\n\nSee common_random.\n\nusage: Usage\nFor interactive/exploratory use. Models should define methods for simulate_data(rng::AbstractRNG, structural_model, θ, ϵ).\n\n\n\nsimulate_data(structural_model, θ)\n\n\nSimulate data, generating ϵ with the default random number generator.\n\nSee common_random.\n\nusage: Usage\nFor interactive/exploratory use. Models should define methods for simulate_data(rng::AbstractRNG, structural_model, θ, ϵ).\n\n\n\n"
},

{
    "location": "general.html#Structural-models-1",
    "page": "General interface",
    "title": "Structural models",
    "category": "section",
    "text": "A structural model can be used to generate data from parameters  and random numbers  using simulate_data. The mapping is not necessarily deterministic even given , but the latter can be used for common random numbers for certain models, to make the mapping continuous, or reduce variance in simulations.When applicable, independent variables (eg covariates) necessary for simulation should be included in the structural model object. It is recommended that a type is defined for each problem, along with methods for simulate_data.simulate_data"
},

{
    "location": "general.html#IndirectLikelihood.common_random",
    "page": "General interface",
    "title": "IndirectLikelihood.common_random",
    "category": "function",
    "text": "common_random(rng, structural_model)\n\n\nReturn common random numbers that can be reused by simulate_data with different parameters.\n\nWhen the model structure does not allow common random numbers, return nothing.\n\nThe first argument is the random number generator.\n\nusage: Usage\nThe user should define a method for this function for each structural_model type.\n\nSee also common_random! for further optimizations.\n\n\n\n"
},

{
    "location": "general.html#IndirectLikelihood.common_random!",
    "page": "General interface",
    "title": "IndirectLikelihood.common_random!",
    "category": "function",
    "text": "common_random!(rng, structural_model, ϵ)\n\n\nUpdate the common random numbers for the model. The semantics is as follows:\n\nit may, but does not need to, change the contents of its second argument,\nthe new common random numbers should be returned regardless.\n\nTwo common usage patterns are\n\nhaving a mutable ϵ, updating that in place, returning ϵ,\ngenerating new ϵ, returning that.\n\nnote: Note\nThe default method falls back to common_random, reallocating with each call. A method for this function should be defined only when allocations can be optimized.\n\n\n\ncommon_random!(rng, problem)\n\n\nReturn a new problem with updated common random numbers.\n\n\n\n"
},

{
    "location": "general.html#common_random_numbers-1",
    "page": "General interface",
    "title": "Common random numbers",
    "category": "section",
    "text": "Common random numbers are a set of random numbers that can be re-used for for simulation with different parameter values.common_random should yield a random value for the random variables (usually an Array or a collection of similar structures).common_randomFor error structures which can be overwritten in place, the user can define common_random! as an optimization.common_random!"
},

{
    "location": "general.html#IndirectLikelihood.MLE",
    "page": "General interface",
    "title": "IndirectLikelihood.MLE",
    "category": "function",
    "text": "MLE(auxiliary_model, data)\n\n\nMaximum likelihood estimate of the parameters for data in model.\n\nWhen ϕ == MLE(auxiliary_model, data), ϕ should maximize\n\nϕ -> loglikelihood(auxiliary_model, data, ϕ)\n\nSee loglikelihood.\n\nMethods should be defined by the user for each auxiliary_model type. A fallback method is defined for nothing as data (infeasible parameters). See also simulate_data.\n\n\n\n"
},

{
    "location": "general.html#IndirectLikelihood.loglikelihood",
    "page": "General interface",
    "title": "IndirectLikelihood.loglikelihood",
    "category": "function",
    "text": "loglikelihood(auxiliary_model, data, ϕ)\n\n\nLog likelihood of data under auxiliary_model with parameters . See MLE.\n\n\n\n"
},

{
    "location": "general.html#Likelihood-1",
    "page": "General interface",
    "title": "Likelihood",
    "category": "section",
    "text": "These methods should be defined for stuctural model types, and accept data from simulate_data.MLE\nloglikelihood"
},

{
    "location": "general.html#IndirectLikelihood.IndirectLikelihoodProblem",
    "page": "General interface",
    "title": "IndirectLikelihood.IndirectLikelihoodProblem",
    "category": "type",
    "text": "IndirectLikelihoodProblem(data, log_prior, structural_model, auxiliary_model, ϵ)\n\nA simple wrapper for an indirect likelihood problem.\n\nThe user should implement simulate_data, MLE, loglikelihood, and common_random.\n\n\n\n"
},

{
    "location": "general.html#IndirectLikelihood.simulate_problem",
    "page": "General interface",
    "title": "IndirectLikelihood.simulate_problem",
    "category": "function",
    "text": "simulate_problem([rng], log_prior, structural_model, auxiliary_model, θ)\n\nInitialize an IndirectLikelihoodProblem with simulated data, using parameters θ.\n\nUseful for debugging and exploration of identification with simulated data.\n\n\n\n"
},

{
    "location": "general.html#Problem-framework-1",
    "page": "General interface",
    "title": "Problem framework",
    "category": "section",
    "text": "IndirectLikelihoodProblem\nsimulate_problem"
},

{
    "location": "general.html#IndirectLikelihood.local_jacobian",
    "page": "General interface",
    "title": "IndirectLikelihood.local_jacobian",
    "category": "function",
    "text": "local_jacobian(problem, θ₀, ω_to_θ; vecϕ)\n\n\nCalculate the local Jacobian of the estimated auxiliary parameters  at the structural parameters =.\n\nω_to_θ maps a vector of reals ω to the parameters θ in the format acceptable to structural_model. It should support ContinuousTransformations.transform and ContinuousTransformations.inverse. See, for example, ContinuousTransformations.TransformationTuple.\n\nvecϕ is a function that is used to flatten the auxiliary parameters to a vector. Defaults to vec_parameters.\n\n\n\n"
},

{
    "location": "general.html#IndirectLikelihood.vec_parameters",
    "page": "General interface",
    "title": "IndirectLikelihood.vec_parameters",
    "category": "function",
    "text": "Return the values of the argument as a vector, potentially (but not necessarily) restricting to those elements that uniquely determine the argument.\n\nFor example, a symmetric matrix would be determined by the diagonal and either half.\n\n\n\n"
},

{
    "location": "general.html#IndirectLikelihood.indirect_loglikelihood",
    "page": "General interface",
    "title": "IndirectLikelihood.indirect_loglikelihood",
    "category": "function",
    "text": "indirect_loglikelihood(auxiliary_model, y, x)\n\nestimate auxiliary_model from summary statistics x using maximum likelihood,\nreturn the likelihood of summary statistics y under the estimated parameters.\n\nUseful for pseudo-likelihood indirect inference, where y would be the observed and x the simulated data.\n\n\n\n"
},

{
    "location": "general.html#Utilities-1",
    "page": "General interface",
    "title": "Utilities",
    "category": "section",
    "text": "local_jacobian\nvec_parameters\nindirect_loglikelihood"
},

{
    "location": "auxiliary.html#",
    "page": "Auxiliary models",
    "title": "Auxiliary models",
    "category": "page",
    "text": ""
},

{
    "location": "auxiliary.html#auxiliary_models-1",
    "page": "Auxiliary models",
    "title": "Auxiliary models",
    "category": "section",
    "text": "Some simple auxiliary models are provided as building blocks."
},

{
    "location": "auxiliary.html#IndirectLikelihood.MvNormalModel",
    "page": "Auxiliary models",
    "title": "IndirectLikelihood.MvNormalModel",
    "category": "type",
    "text": "Model observations as drawn from a multivariate normal distribution. See MvNormalData for summarizing data for estimation and likelihood calculations.\n\n\n\n"
},

{
    "location": "auxiliary.html#IndirectLikelihood.MvNormalData",
    "page": "Auxiliary models",
    "title": "IndirectLikelihood.MvNormalData",
    "category": "type",
    "text": "MvNormalData(n, m, S)\n\nMultivariate normal model summary statistics with n observations, mean m and sample covariance S. Only saves the summary statistics.\n\nusage: Usage\nUse MvNormalData(X, [wv]) to construct from data.\n\n\n\nMvNormalData(X)\n\n\nMultivariate normal summary statistics from observations (each row of X is an observation).\n\n\n\nMvNormalData(X, wv)\n\n\nMultivariate normal summary statistics from observations (each row of X is an observation), with weights.\n\n\n\n"
},

{
    "location": "auxiliary.html#IndirectLikelihood.MvNormalParams",
    "page": "Auxiliary models",
    "title": "IndirectLikelihood.MvNormalParams",
    "category": "type",
    "text": "MvNormalParams(μ, Σ)\n\nParameters for the multivariate normal model x  MvNormal( ).\n\nusage: Usage\nConstruct using MLE(::MvNormalModel, ::MvNormalData).\n\n\n\n"
},

{
    "location": "auxiliary.html#Multivariate-normal-model-1",
    "page": "Auxiliary models",
    "title": "Multivariate normal model",
    "category": "section",
    "text": "MvNormalModel\nMvNormalData\nMvNormalParams"
},

{
    "location": "auxiliary.html#IndirectLikelihood.OLSModel",
    "page": "Auxiliary models",
    "title": "IndirectLikelihood.OLSModel",
    "category": "type",
    "text": "Model data with a scalar- or vector-valued ordinary least squares regression. See OLSData for wrapping data.\n\n\n\n"
},

{
    "location": "auxiliary.html#IndirectLikelihood.OLSData",
    "page": "Auxiliary models",
    "title": "IndirectLikelihood.OLSData",
    "category": "type",
    "text": "OLSData(Y, X)\n\nOrdinary least squares with dependent variable Y and design matrix X.\n\nEither\n\nY is an nm matrix, then Y = X B + E where X is a nk\n\nmatrix, B is a km parameter matrix, and E  N(0 ) is IID with mm variance matrix  (multivariate linear regression), or\n\nY is a length n vector, then Y = X  + , where X is a\n\nnk matrix,  is a parameter vector of k elements, and N(0) where  is the variance of the normal error.\n\nSee also add_intercept.\n\n\n\n"
},

{
    "location": "auxiliary.html#IndirectLikelihood.OLSParams",
    "page": "Auxiliary models",
    "title": "IndirectLikelihood.OLSParams",
    "category": "type",
    "text": "OLSParams(B, Σ)\n\nMaximum likelihood estimated parameters for an OLS regression. See OLSData.\n\n\n\n"
},

{
    "location": "auxiliary.html#IndirectLikelihood.add_intercept",
    "page": "Auxiliary models",
    "title": "IndirectLikelihood.add_intercept",
    "category": "function",
    "text": "add_intercept(X)\n\n\nAdd an intercept to a matrix or vector of covariates.\n\n\n\n"
},

{
    "location": "auxiliary.html#Ordinary-least-squares-model-1",
    "page": "Auxiliary models",
    "title": "Ordinary least squares model",
    "category": "section",
    "text": "OLSModel\nOLSData\nOLSParams\nadd_intercept"
},

{
    "location": "internals.html#",
    "page": "Internals",
    "title": "Internals",
    "category": "page",
    "text": ""
},

{
    "location": "internals.html#IndirectLikelihood.RNG",
    "page": "Internals",
    "title": "IndirectLikelihood.RNG",
    "category": "constant",
    "text": "The default random number generator for methods in this package, used when not specified in the first argument.\n\n\n\n"
},

{
    "location": "internals.html#Internals-1",
    "page": "Internals",
    "title": "Internals",
    "category": "section",
    "text": "IndirectLikelihood.RNG"
},

]}
