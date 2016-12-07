#define ARMA_DONT_USE_CXX11          // Forcefully prevent Armadillo from using C++11 features.
                                     // Must be specified before including RcppArmadillo.h.
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <cvodes/cvodes.h>           // CVODES functions and constants
#include <nvector/nvector_serial.h>  // Serial N_Vector
#include <cvodes/cvodes_dense.h>     // CVDense
#include <datatypes.h>               // RcppSundials data types and helper functions.
#include <algorithm>
#include <string> 
#include <limits> 
#include <array>
#include <vector>
#include <time.h>

using namespace Rcpp; 
using namespace std;
using arma::mat;
using arma::vec;

// [[Rcpp::interfaces(r, cpp)]]

/*
 *
 Functions when the model is written in C++ using the standard library and the Armadillo library
 *
 */

// Interface between cvode integrator and the model function written in Cpp
int cvode_to_Cpp_stl(double t, N_Vector y, N_Vector ydot, void* inputs) {
  // Cast the void pointer back to the correct data structure
  data_Cpp_stl* data = static_cast<data_Cpp_stl*>(inputs);
  // Interpolate the forcings
  vector<double> forcings(data->forcings_data.size());
  if(data->forcings_data.size() > 0) forcings = interpolate_list(data->forcings_data, t);
  // Extract the states from the NV_Ith_S container
  vector<double> states(data->neq);
  for(auto i = 0; i < data->neq ; i++) states[i] = NV_Ith_S(y,i);
  // Run the model
  array<vector<double>, 2> output = data->model(t, states, data->parameters, forcings); 
  // Return the states to the NV_Ith_S
  vector<double> derivatives = output[0];
  for(auto i = 0; i < data->neq; i++)  NV_Ith_S(ydot,i) = derivatives[i];
  return 0; 
}

// Interface cvode with std container Jacobian function.
int cvode_to_Cpp_stl_jac(long int N, double t, N_Vector y, N_Vector fy, DlsMat Jac, 
                   void *inputs, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  // Cast the void pointer back to the correct data structure
  data_Cpp_stl* data = static_cast<data_Cpp_stl*>(inputs);
  // Interpolate the forcings
  vector<double> forcings(data->forcings_data.size());
  if(data->forcings_data.size() > 0) forcings = interpolate_list(data->forcings_data, t);
  // Extract the states from the NV_Ith_S container
  auto neq = data->neq;
  vector<double> states(neq);
  for(auto i = 0; i < neq ; i++) states[i] = NV_Ith_S(y,i);
  // Get the Jacobian
  auto output = data->jacobian(t, states, data->parameters, forcings);
  // Return the DlsMat
  if(output.size() not_eq neq * neq) {
    ::Rf_error("Wrong size of Jacobian matrix.");
  }
  auto it = output.cbegin();
  for(int i = 0; i < neq; ++i) {
    copy(it, it + neq, DENSE_COL(Jac, i));
    advance(it, neq);
  }

  return 0;
}



//' Solve an inital value problem with cvodes.
//'
//' @description Wrapper around the solver cvodes from the Sundials suite.
//'
//' @param times
//'     Numeric vector of time points at which integration results are returned.
//'
//' @param states_
//'     Numeric vector of inital values for all states.
//'
//' @param parameters_
//'     Numeric vector of model parameters values.
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
NumericMatrix wrap_cvodes(NumericVector times, NumericVector states_, 
                        NumericVector parameters_, List forcings_data_, 
                        List settings, SEXP model_, SEXP jacobian_) {
  // Wrap the pointer to the model function with the correct signature                        
  ode_in_Cpp_stl* model =  (ode_in_Cpp_stl *) R_ExternalPtrAddr(model_);
  // Wrap the pointer to the jacobian function with the correct signature                        
  jac_in_Cpp_stl* jacobian =  nullptr;
  if(as<bool>(settings["jacobian"]))
      jacobian = (jac_in_Cpp_stl *) R_ExternalPtrAddr(jacobian_);
  // Store all inputs in the data struct, prior conversion to stl and Armadillo classes
  const auto neq = states_.size();
  vector<double> parameters{as<vector<double>>(parameters_)};
  vector<double> states{as<vector<double>>(states_)};
  vector<mat> forcings_data(forcings_data_.size());
  if(forcings_data_.size() > 0) 
    for(int i = 0; i < forcings_data_.size(); i++)
      forcings_data[i] = as<mat>(forcings_data_[i]);


  /*
   *
   Make a first call to the model to check that everything is ok and retrieve the number of observed variables
   *
   */
  vector<double> forcings(forcings_data.size());
  if(forcings_data.size() > 0) forcings = interpolate_list(forcings_data, times[0]);
  std::array<vector<double>, 2> first_call;
  try {
    first_call = model(times[0], states, parameters, forcings);
  } catch(std::exception &ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }


  // Check length of time derivatives against the information passed through settings
  if(first_call[0].size() != neq) {
    ::Rf_error("Length of time derivatives returned by the model does not coincide with the number of state variables.");
  }

  /*
   *
   Fill up the output matrix with the values for the initial time                    *
   *
   */
  // Get and check number of output states and observations
  auto noutStates = as<long int>(settings["which_states"]);
  auto noutObserved = as<long int>(settings["which_observed"]);
  // This must be checked before comparisons to *.size(), as size_type is
  // unsigned.
  if(noutStates < 0 or noutObserved < 0) {
    ::Rf_error("Negative amount of states or observables requested");
  }

  if(noutStates > neq or noutObserved > first_call[1].size()) {
    ::Rf_error("More states or observations requested than provided by the model");
  }

  if(noutStates == 0 and noutObserved == 0) {
    ::Rf_error("Request at least one state or observable to be returned");
  }

  // Copy all time points, initial states and observations into output.
  // As observations can depend on all states, all states, not only the first
  // noutStates are considered.
  mat output(times.size(), 1 + neq + noutObserved, arma::fill::zeros);
  for(auto h = 0; h < times.size(); ++h) output.at(h, 0) = times[h];
  for(auto h = 0; h < neq; ++h) output.at(0, h + 1) = states[h];
  for(auto h = 0; h < noutObserved; ++h) output.at(0, h + 1 + neq) = first_call[1][h];

  
  /*
   *
   Initialize CVODE and pass all initial inputs
   *
   */
  N_Vector y = nullptr;
  y = N_VNew_Serial(neq);
  copy(states.cbegin(), states.cend(), NV_DATA_S(y));

  void *cvode_mem = nullptr;
  if(as<std::string>(settings["method"]) == "bdf") {
    cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);     
  } else if(as<std::string>(settings["method"]) == "adams"){
    cvode_mem = CVodeCreate(CV_ADAMS, CV_NEWTON);     
  } else {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
    ::Rf_error("Please choose bdf or adams as method");
  }
  
  // Shut up Sundials (errors should not be printed to the screen)
  // FIXME: But they should go somewhere, may be use a dedicated log file.
  int flag = CVodeSetErrFile(cvode_mem, NULL);
  if(flag < CV_SUCCESS) {
   if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
   if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}       
   ::Rf_error("Error in the CVodeSetErrFile function");
  }
  
  // Initialize the Sundials solver. Here we pass initial N_Vector, the interface function and the initial time
  flag = CVodeInit(cvode_mem, cvode_to_Cpp_stl, times[0], y);
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}     
    ::Rf_error("Error in the CVodeInit function");
  }
  
  // Tell Sundials the tolerance settings for error control
  Rcpp::NumericVector abstol = settings["atol"]; 
  if(abstol.size() > 1) {
    N_Vector Nabstol = nullptr;
    Nabstol = N_VNew_Serial(neq);
    for(int i = 0; i < neq; i++) {
      NV_Ith_S(Nabstol,i) = abstol[i];
    }
    flag = CVodeSVtolerances(cvode_mem, settings["rtol"], Nabstol);
  } else {
    flag = CVodeSStolerances(cvode_mem, settings["rtol"], settings["atol"]);    
  }

  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSStolerances function");
  }
  
  // Tell Sundials the number of state variables, so that I can allocate memory for the linear solver
  flag = CVDense(cvode_mem, neq);
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVDense function");
  }

  // Give Sundials a pointer to the struct where all the user data is stored. It will be passed (untouched) to the interface as void pointer
  data_Cpp_stl data_model{parameters, forcings_data, neq, model, jacobian};
  flag = CVodeSetUserData(cvode_mem, &data_model);
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetUserData function");
  }
  
  // If we want to provide our own Jacobian, set the interface function to Sundials
  if(as<bool>(settings["jacobian"])) {
    flag = CVDlsSetDenseJacFn(cvode_mem, cvode_to_Cpp_stl_jac);
    if(flag < CV_SUCCESS) {
      if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
      if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
      ::Rf_error("Error in the CVDlsSetDenseJacFn function");
    }
  }

  // Set maximum number of steps
  flag = CVodeSetMaxNumSteps(cvode_mem, settings["maxsteps"]);
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetUserData function");
  }
  
  // Set maximum order of the integration
  flag = CVodeSetMaxOrd(cvode_mem, settings["maxord"]); 
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetMaxOrd function");
  }
  
  // Set the initial step size
  flag = CVodeSetInitStep(cvode_mem, settings["hini"]);  
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetInitStep function");
  }
  
  // Set the minimum step size
  flag = CVodeSetMinStep(cvode_mem, settings["hmin"]);  
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetMinStep function");
  }
  
  // Set the maximum step size
  flag = CVodeSetMaxStep(cvode_mem, settings["hmax"]);  
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetMaxStep function");
  }
  
  // Set the maximum number of error test fails
  flag = CVodeSetMaxErrTestFails(cvode_mem, settings["maxerr"]);  
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetMaxErrTestFails function");
  }
  
  // Set the maximum number of nonlinear iterations per step
  flag = CVodeSetMaxNonlinIters(cvode_mem, settings["maxnonlin"]);  
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetMaxNonlinIters function");
  }
  
  // Set the maximum number of convergence failures
  flag = CVodeSetMaxConvFails(cvode_mem, settings["maxconvfail"]);   
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetMaxConvFails function");
  }
  
  // Set stability limit detection
  flag = CVodeSetStabLimDet(cvode_mem, as<bool>(settings["stability"]));  
  if(flag < CV_SUCCESS) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);} 
    ::Rf_error("Error in the CVodeSetStabLimDet function");
  }
  

  

  // FIXME: For large noutStates, it might be faster to iterate over
  // ydata = NV_DATA_S(y) instead of using NV_Ith_S(y,i).
  /*
   *
   Main time loop. Each timestep call cvode. Handle exceptions and fill up output
   *
   */
  // Preparations
  auto checkPositive = as<bool>(settings["positive"]);
  auto zero = as<double>(settings["minimum"]);
  double t = times[0];

  try {
    for(int i = 1; i < times.size(); i++) {
      flag = CVode(cvode_mem, times[i], y, &t, CV_NORMAL);
      if(checkPositive) {
        for(auto h = 0; h < neq; h++) {
         if(NV_Ith_S(y,h) < zero) {
           Rcout << "The state variable at position " << h + 1 << " became smaller than minimum: " << NV_Ith_S(y,h) << " at time: " << times[i] << '\n';
           if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
           if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
           ::Rf_error("At least one of the states became smaller than minimum");
         }
        }
      }
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
        }
      }

      // Write states to output
      for(auto h = 0; h < neq; ++h) output(i, h + 1) = NV_Ith_S(y,h);
    }
  } catch(std::exception &ex) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
    forward_exception_to_r(ex);
  } catch(...) {
    if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
    if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}
    ::Rf_error("c++ exception (unknown reason)");
  }


  // If we have observed variables we call the model function again
  if(noutObserved > 0 && flag >= 0.0) {
    for(unsigned i = 1; i < times.size(); i++) {
      // Get forcings values at time 0.
      if(forcings_data.size() > 0) forcings = interpolate_list(forcings_data, times[i]);
      // Get the simulate state variables
      for(auto j = 0; j < neq; j++) states[j] = output(i,j + 1);
      // Call the model function to retrieve total number of outputs and initial values for derived variables
      std::array<vector<double>, 2> model_call  = model(times[i], states, parameters, forcings);
      // Derived variables already stored by the interface function
      for(auto h = 0; h < noutObserved; ++h) output(i, h + 1 + neq) = model_call[1][h];
    }
  }
              
  // De-allocate the N_Vector and the cvode_mem structures
  if(y == nullptr) {free(y);} else {N_VDestroy_Serial(y);}
  if(cvode_mem == nullptr) {free(cvode_mem);} else {CVodeFree(&cvode_mem);}

  // Subset output for time, noutStates, and noutObserved.
  vector<unsigned int> idxAux(1 + noutStates + noutObserved, 0);
  iota(idxAux.begin() + 1, idxAux.begin() + 1 + noutStates, 1);
  iota(idxAux.begin() + 1 + noutStates, idxAux.end(), neq + 1);
  arma::uvec idx(idxAux);

  return wrap(static_cast<mat>(output.cols(idx)));
}

//' Allows calling the model that calculates the time derivatives
//' @export
// [[Rcpp::export]]
List cvode_calc_derivs(SEXP model_, NumericVector t, NumericVector states, 
                          NumericVector parameters, List forcings_data_) {
   // Wrap the pointer to the model function with the correct signature                        
  ode_in_Cpp_stl* model =  (ode_in_Cpp_stl *) R_ExternalPtrAddr(model_); 
  // Interpolate the forcings
  vector<mat> forcings_data(forcings_data_.size());
  if(forcings_data_.size() > 0) 
    for(int i = 0; i < forcings_data_.size(); i++)
      forcings_data[i] = as<mat>(forcings_data_[i]);
  vector<double> forcings(forcings_data.size());
  if(forcings_data.size() > 0) forcings = interpolate_list(forcings_data, t[0]);
  // Call the model
  array<vector<double>, 2> output = model(t[0], as<vector<double>>(states),
                                          as<vector<double>>(parameters),
                                          forcings);
  // return the output as a list
  return List::create(_["Derivatives"] = wrap(output[0]),
                      _["Observed"] = wrap(output[1]));
}

//' Allows calling the function to calculate the Jacobian matrix of the model
//' @export
// [[Rcpp::export]]
NumericMatrix cvode_calc_jac(SEXP jacobian_, NumericVector t, NumericVector states, 
                          NumericVector parameters, List forcings_data_) {
   // Wrap the pointer to the model function with the correct signature                        
  jac_in_Cpp_stl* jacobian = (jac_in_Cpp_stl *) R_ExternalPtrAddr(jacobian_);
  // Interpolate the forcings
  vector<mat> forcings_data(forcings_data_.size());
  if(forcings_data_.size() > 0) 
    for(int i = 0; i < forcings_data_.size(); i++)
      forcings_data[i] = as<mat>(forcings_data_[i]);
  vector<double> forcings(forcings_data.size());
  if(forcings_data.size() > 0) forcings = interpolate_list(forcings_data, t[0]);
  // Call the model
  arma::mat output = jacobian(t[0], as<vector<double>>(states),
                                          as<vector<double>>(parameters),
                                          forcings);
  // return the output as a list
  return wrap(output);
} 
