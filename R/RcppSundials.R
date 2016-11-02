#' Sundials for R
#'
#' Wrapper around Sundials numerical library using Rcpp and RcppArmadillo
#'
#' Sundials is a SUite of Nonlinear and DIfferential/ALgebraic Equation Solvers.
#' This packages interfaces R with the solver cvodes and ida. cvodes is a solver
#' for stiff and nonstiff systems of ordinary linear equations, ida solves
#' differential-algebraic equation systems. Sundials files are included such
#' that no local installation of Sundials is required.
#'
#' Containers from the C++ stl or the Armadillo library must be used to
#' formulate the system of equations which is to be solved.
#'
#' In addition to the main integration function, a couple of other C++ function
#' are exported, such that the user can call these functions directly, without
#' going through R. That way, expensive tasks such as parameter optimization or
#' sensitivity analyses may be performed faster.
#'
#' The interface \pkg{RcppSundials} is incompatible with the interface provided
#' by \pkg{deSolve}. The main different is that interpolation of external
#' forcings is performed by RcppSundials whereas deSolve does not. Also, inputs
#' cannot be passed as global variables, in contrast to deSolve when models are
#' formulated in the C or Fortran.
#'
"_PACKAGE"
