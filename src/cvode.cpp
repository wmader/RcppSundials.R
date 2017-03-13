#include <string>
#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <RcppArmadillo.h>
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <cvodes/cvodes_dense.h>
#include <datatypes.h>
#include <interfaces.h>
#include <support.h>



// [[Rcpp::interfaces(r, cpp)]]


//' Solve an inital value problem with cvodes.
//'
//' @description Wrapper around the solver cvodes from the Sundials suite.
//'
//' @param times
//'     Numeric vector of time points at which integration results are returned.
//'
//' @param states_
//'     Numeric vector of inital values for states.
//'
//' @param parameters_
//'     Numeric vector of model parameters values.
//'
//' @param initSens_
//'     Numeric vector of inital values for sensitivities.
//'
//' @param forcings_data_
//'     List of forcings acting on the system.
//'
//' @param settings
//'     List of setting passed to cvodes. For a detailed documentation of the
//'     supported setting please check the
//'     \href{http://computation.llnl.gov/projects/sundials/sundials-software}{Sundials homepage}.
//'     Supported settings are
//'     \describe{
//'     \item{\code{"jacobian"}, bool.}{
//'     For \code{"jacobian" = TRUE}, a function returning the Jacobian matrix
//'     of the system must be provided by \option{jacobian_}.}
//'
//'     \item{\code{"method"}, string, can be \code{"bdf"} or \code{"adams"}.}{
//'     The integration method used. For "bdf" \code{CVodeCreate(CV_BDF, CV_NEWTON)}
//'     is called, for "adams" \code{CVodeCreate(CV_ADAMS, CV_NEWTON)}.}
//'
//'     \item{\code{"atol"}, a scalar or a vector.}{
//'     Specifies the absolute integration tolerance. If "atol" is scalar, each
//'     state is integrated with the same absolute tolerance. If the absolute
//'     error tolerance needs to be different for each state, "atol" can be a
//'     vector holding the tolerance for each state.}
//'
//'     \item{\code{"rtol"}, scalar.}{
//'     Relative integration error tolerance.}
//'
//'     \item{\code{"which_states"}, vector.}{
//'     Return the first \code{"which_states"}. If the model has \code{N} states,
//'     \code{which_states <= N} allows to dicard all states
//'     \code{> which_states}
//'     }
//'
//'     \item{\code{"which_observed"}, vector.}{
//'     Same as \code{"which_states"}, but for observables.
//'     }
//'
//'     \item{\code{"maxsteps" = 500}, scalar.}{
//'     Maximum number of internal steps allowed to reach the next output time.
//'     While not recommended, this test can be disabled by passing
//'     \code{"maxsteps" < 0}.}
//'
//'     \item{\code{"maxord" = 12 (adams) or 5 (bdf)}, scalar.}{
//'     Maximum order of the linear multistep method. Can only be set to smaller
//'     values than default.}
//'
//'     \item{\code{"hini" = "estimated"}, scalar.}{
//'     Inital step size.}
//'
//'     \item{\code{"hmin" = 0.0}, scalar.}{
//'     Minimum absolute step size.}
//'
//'     \item{\code{"hmax" = infinity}, scalar.}{
//'     Maximum absolute step size.}
//'
//'     \item{\code{"maxerr"} = 7, scalar.}{
//'     Permitted maximum number of failed error test per step.}
//'
//'     \item{\code{"maxnonlin" = 3}, scalar.}{
//'     Permitted nonlinear solver iterations per step.}
//'
//'     \item{\code{"maxconvfail" = 10}, scalar.}{
//'     Permitted convergence failures of the nonlinear solver per step.}
//'
//'     \item{\code{"stability"} = FALSE, bool.}{
//'     Stability limit detection for the "bdf" method.}
//'
//'     \item{\code{"positive"}, bool.}{
//'     Issue an error (and abort?) in case a state becomes smaller than
//'     \option{"minimum"}.}
//'
//'     \item{\code{"minimum"}, scalar.}{
//'     Lower bound below which a state is assumed negative and reported, in
//'     case \option{\code{"positive" = TRUE}}.}
//'
//'     \item{\code{"sensitivities"} = FALSE, bool.}{
//'     Integrate sensitivities of the dynamic system.}
//'     }
//'
//' @param model_ The address of the ode model. The address is obtained as the
//'     attribute \code{address} of \code{\link[base]{getNativeSymbolInfo}}. The
//'     signature of the model function must comply to
//'
//'     \code{std::array<std::vector<double>, 2> (const double& t, const std::vector<double>& states,
//'     const std::vector<double>& parameters, const std::vector<double>& forcings)}
//'
//'     Return vector \code{std::array<std::vector<double>, 2>}
//'     \enumerate{
//'     \item
//'     First dimension holds the increments for all
//'     states.
//'     \item
//'     Second dimension holds the observed state. Not sure what these are.
//'     }
//'
//'     Argument list
//'     \code{(const double& t, const std::vector<double>& states,
//'     const std::vector<double>& parameters, const std::vector<double>& forcings)}
//'     \describe{
//'     \item{t}{
//'     Most probably the requested time point, but I am not totally sure.}
//'
//'     \item{states}{
//'     Vector of current state values.}
//'
//'     \item{parameters}{
//'     Vector of parameters values.}
//'
//'     \item{forcings}{
//'     Vector of forcings acting on the model.}
//'     }
//'
//' @param jacobian_
//'     The address of the function which returns the Jacobian matrix of the
//'     model. Again, this address is the attribute \code{address} obtained
//'     from the call to \code{\link[base]{getNativeSymbolInfo()}}. The function
//'     must have the signature
//'     \code{arma::mat (const double& t, const std::vector<double>& states, const std::vector<double>& parameters, const std::vector<double>& forcings)}
//'     Returned is the Jacobian matrix as an \code{arma::mat} from the
//'     Armadillo package.
//'
//'     The list of arguments is the same as for \option{model_}.
//'
//' @details This function sets up the cvodes integrator and loop over the
//'     vector \option{times} of requested time points. On success, the states
//'     of the system are returend for these time points.
//'
//'     Write something about these observations, once you got a hold of them.
//'
//' @return Matrix with nrow = (no. timepoints) and
//'     ncol = (no. states + no. observed + [(no. states)x(no. parameters)]).
//'     [(no. states)x(no. parameters)] is only returned if sensitivity equations
//'     are calculated.
//'
//'     \describe{
//'     \item{First column}{
//'     Integration time points as given in \option{times}.}
//'
//'     \item{Column 2 to no. of states + 1}{
//'     The state for the respective time point.}
//'
//'     \item{no. of states + 1 to number of states + 1 + n. of observations}{
//'     Observation for the respective time point.}
//'     }
//'
//' @author Alejandro Morales, \email{morales.s.alejandro@@gmail.com}
//'
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix wrap_cvodes(Rcpp::NumericVector times,
                                Rcpp::NumericVector states_,
                                Rcpp::NumericVector parameters_,
                                Rcpp::NumericVector initSens_,
                                Rcpp::List forcings_data_,
                                Rcpp::List settings,
                                SEXP model_, SEXP jacobian_, SEXP sens_) {
  // Cast function pointers
  statesRHS* model = reinterpret_cast<statesRHS*>(R_ExternalPtrAddr(model_));

  statesJacRHS* jacobian = nullptr;
  auto isJac = Rcpp::as<bool>(settings["jacobian"]);
  if(isJac) jacobian = reinterpret_cast<statesJacRHS*>(R_ExternalPtrAddr(jacobian_));

  sensitivitiesRHS* sensitivities = nullptr;
  auto isSens = Rcpp::as<bool>(settings["sensitivities"]);
  if(isSens) sensitivities = reinterpret_cast<sensitivitiesRHS*>(R_ExternalPtrAddr(sens_));



  // Convert input to standard containers
  auto stateInits(Rcpp::as<std::vector<double>>(states_));
  auto parameters(Rcpp::as<std::vector<double>>(parameters_));



  // Test model evaluation
  const int neq = stateInits.size();
  checkModel(times[0], neq, stateInits, parameters, model);



  // Store initial states in output matrix
  // As armadillo is column-major, output containers are allocated such that
  // each column refers to one time point.
  const int nTimepoints = times.size();
  arma::mat outputStates(neq, nTimepoints);
  std::copy(stateInits.cbegin(), stateInits.cend(), outputStates.begin_col(0));



  //////////////////////
  // Initialize CVODE //
  //////////////////////

  // Copy initial states into cvode state container y.
  N_Vector y = N_VNew_Serial(neq);
  std::copy(stateInits.cbegin(), stateInits.cend(), NV_DATA_S(y));

  void* cvode_mem = nullptr;
  UserDataIVP data_model{neq, parameters, model, jacobian, sensitivities};

  try {
    // Instantiate a CVODES solver object
    cvode_mem = createCVodes(settings);

    // Set error output file
    // FIXME: Errors should go somewhere. Right now, they are simply discarded.
    int flag = CVodeSetErrFile(cvode_mem, nullptr);
    cvSuccess(flag, "Error: Setting error output file.");


    // Initialize CVODES solver object
    flag = CVodeInit(cvode_mem, CVRhsFnIf, times[0], y);
    cvSuccess(flag, "Could not Initialize CVODES solver object.");

    // Set absolute and relative tolerance for integration
    flag = CVodeSStolerances(cvode_mem, settings["rtol"], settings["atol"]);
    cvSuccess(flag, "Error on setting integration tolerance.");

    // Select linear solver CVDENSE
    flag = CVDense(cvode_mem, neq);
    cvSuccess(flag, "Could not set dense linear solver.");

    // Attache user data to CVODES memory block
    flag = CVodeSetUserData(cvode_mem, &data_model);
    cvSuccess(flag, "Failure: Attach user data.");

    // Do we supply equations for the Jacobian? If so, set them.
    if(Rcpp::as<bool>(settings["jacobian"])) {
      flag = CVDlsSetDenseJacFn(cvode_mem, CVDlsDenseJacFnIf);
      cvSuccess(flag, "Failure: Setup user-supplied Jacobian function.");
    }

    // Set maximum number of steps taken by the solver to reach next output time
    flag = CVodeSetMaxNumSteps(cvode_mem, settings["maxsteps"]);
    cvSuccess(flag, "Could not set maximum number of steps.");

    // Set maximum order of the linear multistep method
    flag = CVodeSetMaxOrd(cvode_mem, settings["maxord"]);
    cvSuccess(flag, "Error: Specifying maximum order of linear multistep method. ");

    // Set initial step size
    flag = CVodeSetInitStep(cvode_mem, settings["hini"]);
    cvSuccess(flag, "Error: Setting initial step size.");

    // Set minimum step size
    flag = CVodeSetMinStep(cvode_mem, settings["hmin"]);
    cvSuccess(flag, "Error:S etting minimum step size.");

    // Set the maximum step size
    flag = CVodeSetMaxStep(cvode_mem, settings["hmax"]);
    cvSuccess(flag, "Error: Setting maximum step size.");

    // Set the maximum number of error test fails per step
    flag = CVodeSetMaxErrTestFails(cvode_mem, settings["maxerr"]);
    cvSuccess(flag, "Error: Setting error test fails.");

    // Set the maximum number of nonlinear iterations per step
    flag = CVodeSetMaxNonlinIters(cvode_mem, settings["maxnonlin"]);
    cvSuccess(flag, "Error: Setting maximum number of nonlinear solver iterations.");

    // Set the maximum number of nonlinear solver convergence failures per step
    flag = CVodeSetMaxConvFails(cvode_mem, settings["maxconvfail"]);
    cvSuccess(flag, "Error: Setting maximum number of nonlinear solver convergence failures.");

    // Should BDF stability limit detection
    flag = CVodeSetStabLimDet(cvode_mem, Rcpp::as<bool>(settings["stability"]));
    cvSuccess(flag, "Error: Setting BDF stability limit detection.");
  }
  catch(std::exception &ex) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
    Rcpp::stop(ex.what());
  } catch(...) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
    Rcpp::stop("Unknown error on initializing the CVODES ");
  }



  //////////////////////////////
  // Initialize sensitivities //
  //////////////////////////////

  // Prepare
  const int Ns = neq + parameters.size();
  const int nPar = parameters.size();
  const int nSens = neq * (neq + nPar);
  N_Vector* yS = nullptr;
  arma::mat outputSensitivities;

  if(isSens) {
    // Set cvodes and output container to correct size
    yS = N_VCloneVectorArray_Serial(Ns, y);
    outputSensitivities.set_size(nSens, nTimepoints);

    // Convert sensitivity initials to standard containers
    std::vector<double> sensitivities{Rcpp::as<std::vector<double>>(initSens_)};

    // Copy initial sensitivities to sensitivity output container
    std::copy(sensitivities.begin(), sensitivities.end(), outputSensitivities.begin_col(0));

    // Initialize cvodes sensitivity container yS
    auto it = sensitivities.cbegin();
    for(int i = 0; i < Ns; ++i) {
      std::copy(it, it + neq, NV_DATA_S(yS[i]));
      advance(it, neq);
    }

    try {
      // Switch on sensitivity calculation in cvodes
      int flag = CVodeSensInit(cvode_mem, Ns, CV_SIMULTANEOUS, CVSensRhsFnIf, yS);
      cvSuccess(flag, "Error: Switch on sensitivities.");

      // FIXME: Use scalar tolerances.
      flag = CVodeSensEEtolerances(cvode_mem);
      cvSuccess(flag, "Error: Setting sensitivity tolerances.");
    } catch(std::exception &ex) {
      if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
      if(yS == nullptr) {free(yS);} else {N_VDestroyVectorArray_Serial(yS, Ns);}
      if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
      Rcpp::stop(ex.what());
    } catch(...) {
      if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
      if(yS == nullptr) {free(yS);} else {N_VDestroyVectorArray_Serial(yS, Ns);}
      if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
      Rcpp::stop("C++ exception (unknown reason)");
    }
  }



  ////////////////////
  // Main time loop //
  ////////////////////

  try {
    // Prepare
    double tretStates = 0;
    double tretSensitivities = 0;

    // In each time-step, solutions are advanced by calling CVode
    for(int t = 1; t < nTimepoints; ++t) {
      int flag = CVode(cvode_mem, times[t], y, &tretStates, CV_NORMAL);
      if(flag < CV_SUCCESS) {
        switch(flag) {
          case CV_TOO_MUCH_WORK:
            throw std::runtime_error("The solver took mxstep internal steps but could not reach tout."); break;
          case CV_TOO_MUCH_ACC:
            throw std::runtime_error("The solver could not satisfy the accuracy demanded by the user for some internal step"); break;  
          case CV_ERR_FAILURE:
            throw std::runtime_error("Error test failures occured too many times during one internal time step or minimum step size was reached"); break;    
          case CV_CONV_FAILURE:
            throw std::runtime_error("Convergence test failures occurred too many times during one internal time step or minimum step size was reached."); break;  
          case CV_LINIT_FAIL:
            throw std::runtime_error("The linear solver’s initialization function failed."); break; 
          case CV_LSETUP_FAIL:
            throw std::runtime_error("The linear solver’s setup function failed in an unrecoverable manner"); break; 
          case CV_LSOLVE_FAIL:
            throw std::runtime_error("The linear solver’s solve function failed in an unrecoverable manner"); break;  
          case CV_RHSFUNC_FAIL:
            throw std::runtime_error("The right hand side function failed in an unrecoverable manner"); break;
          case CV_FIRST_RHSFUNC_ERR:
            throw std::runtime_error("The right-hand side function failed at the first call."); break;    
          case CV_REPTD_RHSFUNC_ERR:
            throw std::runtime_error("The right-hand side function had repeated recoverable errors."); break;   
          case CV_UNREC_RHSFUNC_ERR:
            throw std::runtime_error("The right-hand side function had a recoverable errors but no recovery is possible."); break; 
          case CV_BAD_T:
            throw std::runtime_error("The time t is outside the last step taken."); break; 
          case CV_BAD_DKY:
            throw std::runtime_error("The output derivative vector is NULL."); break;   
          case CV_TOO_CLOSE:
            throw std::runtime_error("The output and initial times are too close to each other."); break;
          case CV_ILL_INPUT:
            throw std::runtime_error("Input to CVode or to its solver illegal or missing."); break;
          default:
            throw std::runtime_error(std::string("CVodes error code: ") + std::to_string(flag)); break;
        }
      }


      //////////////////////
      // Read out results //
      //////////////////////

      // Copy states to output container
      std::copy(NV_DATA_S(y), NV_DATA_S(y) + neq, outputStates.begin_col(t));

      // Copy sensitivities to output container
      flag = CVodeGetSens(cvode_mem, &tretSensitivities, yS);
      if(flag < CV_SUCCESS) {
        throw std::runtime_error("Error in CVodeGetSens: Could not extract sensitivities.");
      } else {
        auto itOutSens = outputSensitivities.begin_col(t);
        for(auto j = 0; j < Ns; ++j) {
          auto sensData = NV_DATA_S(yS[j]);
          std::copy(sensData, sensData + neq, itOutSens);
          std::advance(itOutSens, neq);
        }
      }

    }
  } catch(std::exception &ex) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(yS == nullptr) {free(yS);} else {N_VDestroyVectorArray_Serial(yS, Ns);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
    Rcpp::stop(ex.what());
  } catch(...) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(yS == nullptr) {free(yS);} else {N_VDestroyVectorArray_Serial(yS, Ns);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
    Rcpp::stop("C++ exception (unknown reason)");
  }


//   // If we have observed variables we call the model function again
//   if(noutObserved > 0 && flag >= 0.0) {
//     for(unsigned int i = 1; i < nTimepoints; ++i) {
//       // Get the simulate state variables
//       for(auto j = 0; j < neq; j++) stateInits[j] = output(i,j + 1);
//       // Call the model function to retrieve total number of outputs and initial values for derived variables
//       std::array<std::vector<double>, 2> model_call  = model(times[i], stateInits, parameters);
//       // Derived variables already stored by the interface function
//       for(auto h = 0; h < noutObserved; ++h) output(i, h + 1 + neq) = model_call[1][h];
//     }
//   }



  ////////////////////////
  // Cleanup and return //
  ////////////////////////

  // Free our resources
  if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
  if(yS == nullptr) {free(yS);} else {N_VDestroyVectorArray_Serial(yS, Ns);}
  if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}

//   // Subset output for time, noutStates, and noutObserved.
//   std::vector<unsigned int> idxAux(1 + noutStates + noutObserved, 0);
//   iota(idxAux.begin() + 1, idxAux.begin() + 1 + noutStates, 1);
//   iota(idxAux.begin() + 1 + noutStates, idxAux.end(), neq + 1);
//   arma::uvec idx(idxAux);
//
//   if(isSens) {
//     return Rcpp::wrap(static_cast<arma::mat>(arma::join_horiz(output.cols(idx), outSensitivities)));
//   } else {
//     return Rcpp::wrap(static_cast<arma::mat>(output.cols(idx)));
//   }


  // Prepare output and return
  arma::mat outputTime(nTimepoints, 1);
  std::copy(times.begin(), times.end(), outputTime.begin_col(0));

  arma::inplace_trans(outputStates);
  arma::inplace_trans(outputSensitivities);

  arma::mat output = arma::join_horiz(arma::join_horiz(outputTime, outputStates), outputSensitivities);

  return Rcpp::wrap(static_cast<arma::mat>(output));
}



/////////////////////////////////
// Discard >n states on return //
/////////////////////////////////

// // Get and check number of output states and observations
// auto noutStates = Rcpp::as<long int>(settings["which_states"]);
// auto noutObserved = Rcpp::as<long int>(settings["which_observed"]);
// // This must be checked before comparisons to *.size(), as size_type is
// // unsigned.
// if(noutStates < 0 or noutObserved < 0) {
//   ::Rf_error("Negative amount of states or observables requested");
// }
//
// if(noutStates > neq or noutObserved > first_call[1].size()) {
//   ::Rf_error("More states or observations requested than provided by the model");
// }
//
// if(noutStates == 0 and noutObserved == 0) {
//   ::Rf_error("Request at least one state or observable to be returned");
// }
