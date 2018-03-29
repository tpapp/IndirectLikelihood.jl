# Introduction

This package implements a general framework for *indirect inference* using likelihood-based methods. It is composed of two parts:

1. a very light [general interface](@ref general_interface) for organizing an indirect inference problem, mostly for avoiding repeated code,

2. some simple [auxiliary models](@ref auxiliary_models) that can be used as building blocks (the interface is of course general enough to admit arbitrary models).

This package assumes that you are familiar with the concept of indirect inference. If not, Smith (2008) is a good starting point, then see Drovandi et al (2015) for a survey on Bayesian methods, and Gallant & McCulloch (2009) for the particular method implemented by this package.

In a nutshell, indirect inference is useful when you know how to *simulate* data from a given set of parameters for some *structural model*, but the problem is too complex to admit a direct likelihood-based representation. Instead, an *auxiliary model* is estimated on the actual and simulated data, and the distance between the two parameter estimates is minimized under some metric.

The method of Gallant & McCulloch (2009) is particularly elegant because it fits into a likelihood-based framework, and induces a natural distance metric. Methods used in frequentist statistics/econometrics often rely on a weighting matrix instead, which needs to be estimated too, but it is not needed for this framework.

## References

- Drovandi, C. C., Pettitt, A. N., Lee, A., & others, (2015). Bayesian indirect
  inference using a parametric auxiliary model. *Statistical Science*, 30(1),
  72–95. [(preprint)](https://arxiv.org/pdf/1505.03372)

- Gallant, A. R., & McCulloch, R. E. (2009). *On the determination of general
  scientific models with application to asset pricing*. Journal of the American
  Statistical Association, 104(485), 117–131. [(preprint)](http://www.rob-mcculloch.org/some_papers_and_talks/papers/published/gsm_gallant_mcculloch.pdf) [(JSTOR)](http://www.jstor.org/stable/40591904)

- Smith, A. Indirect inference. *The New Palgrave Dictionary of Economics, 2nd Edition (forthcoming)* (2008).
