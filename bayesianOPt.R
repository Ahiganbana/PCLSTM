library("rBayesianOptimization")
# Example 1: Optimization
## Set Pred = 0, as placeholder
Test_Fun <- function(x) {
list(Score = x^2,
Pred = 0)
}
## Set larger init_points and n_iter for better optimization result
OPT_Res <- BayesianOptimization(Test_Fun,
bounds = list(x = c(1, 3)),
init_points = 2, n_iter = 1,
acq = "ucb", kappa = 2.576, eps = 0.0,
verbose = TRUE)
