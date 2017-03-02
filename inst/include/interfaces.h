/*
 * Interface between cvodes, C, and the systems of ordinary differential
 * equations which are formulated in C++.
 *
 * Interface functions:
 * - cvode_to_Cpp_stl
 *   Interface to the odes of states.
 *
 * - cvode_to_Cpp_stl_jac
 *   Interface to the jacobian of states.
 *
 * - CVSensRhsFnIf
 *   Interface to the odes of sensitivities.
 *
 */

#ifndef _RCPPSUNDIALSINTERFACES_H
#define _RCPPSUNDIALSINTERFACES_H

// #define ARMA_DONT_USE_CXX11
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <cvodes/cvodes.h>
#include <cvodes/cvodes_dense.h>
#include <nvector/nvector_serial.h>
#include <vector>
#include <array>
#include <algorithm>
#include <iterator>


using namespace Rcpp;
using arma::mat;
using arma::vec;


// Interface between cvode integrator and the model function written in Cpp
int cvode_to_Cpp_stl(double t, N_Vector y, N_Vector ydot, void* inputs) {
  // Cast the void pointer back to the correct data structure
  data_Cpp_stl* data = static_cast<data_Cpp_stl*>(inputs);
  // Interpolate the forcings
  std::vector<double> forcings(data->forcings_data.size());
  if(data->forcings_data.size() > 0) forcings = interpolate_list(data->forcings_data, t);
  // Extract the states from the NV_Ith_S container
  std::vector<double> states(data->neq);
  for(auto i = 0; i < data->neq ; i++) states[i] = NV_Ith_S(y,i);
  // Run the model
  std::array<std::vector<double>, 2> output = data->model(t, states, data->parameters, forcings);
  // Return the states to the NV_Ith_S
  std::vector<double> derivatives = output[0];
  for(auto i = 0; i < data->neq; i++)  NV_Ith_S(ydot,i) = derivatives[i];
  return 0;
}

// Interface cvode with std container Jacobian function.
int cvode_to_Cpp_stl_jac(long int N, double t, N_Vector y, N_Vector fy, DlsMat Jac,
                   void *inputs, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  // Cast the void pointer back to the correct data structure
  data_Cpp_stl* data = static_cast<data_Cpp_stl*>(inputs);
  // Interpolate the forcings
  std::vector<double> forcings(data->forcings_data.size());
  if(data->forcings_data.size() > 0) forcings = interpolate_list(data->forcings_data, t);
  // Extract the states from the NV_Ith_S container
  auto neq = data->neq;
  std::vector<double> states(neq);
  for(auto i = 0; i < neq ; i++) states[i] = NV_Ith_S(y,i);
  // Get the Jacobian
  auto output = data->jacobian(t, states, data->parameters, forcings);
  // Return the DlsMat
  if(output.size() not_eq neq * neq) {
    ::Rf_error("Wrong size of Jacobian matrix.");
  }
  auto it = output.cbegin();
  for(int i = 0; i < neq; ++i) {
    std::copy(it, it + neq, DENSE_COL(Jac, i));
    std::advance(it, neq);
  }

  return 0;
}



/** Interfacing cpp sensitivity description with cvodes.
 *
 * This function's signature matches cvodes requested signature for the
 * function which computes the sensitivity right-hand side for all
 * sensitivities. This function it documented in cvs_guide.pdf under the
 * keyword CVSensRhsFn.
 *
 * Internally, this function calls userData->sensitivities which is the
 * user-supplied function computing the sensitivity right-hand side for all
 * sensitivities formulated in C++.
 *
 * Parameters are named according to cvs_guide.pdf, documentation is copied.
 * @param Ns[in] Number of parameters for which sensitivities are computed.
 * @param t[in] The current value of the independent variable.
 * @param y[in] The current value of the state vector.
 * @param ydot[in] The current value of the right-hand side of the state equations.
 * @param yS[in] The current values of the sensitivity vectors.
 * @param ySdot[out] The output of CVSensRhsFn. On exit it must contain the sensitivity right-hand side vectors.
 * @param user_data[in] A pointer to user data, the same as the user data parameter passed to CVodeSetUserData.
 * @param tmp1, tmp2[in] N Vectors of length N which can be used as temporary storage.
 */
int CVSensRhsFnIf(int Ns, double t, N_Vector y, N_Vector ydot,
                  N_Vector *yS, N_Vector *ySdot,
                  void *user_data,
                  N_Vector tmp1, N_Vector tmp2) {
  data_Cpp_stl* userData = static_cast<data_Cpp_stl*>(user_data);

  // Transform N_Vector to std::vector
  const auto& parameters = userData->parameters;
  const auto neq = userData->neq;
  const auto npar = parameters.size();

  // Copy current states
  std::vector<double> states(neq);
  const auto statesData = NV_DATA_S(y);
  std::copy(statesData, statesData + neq, states.begin());

  // Copy current sensitivities
  std::vector<double> sensCur(Ns * neq);
  auto itSensCur = sensCur.begin();
  for(auto i = 0; i < Ns; ++i) {
    const auto sensitivityData = NV_DATA_S(yS[i]);
    std::copy(sensitivityData, sensitivityData + neq, itSensCur);
    std::advance(itSensCur, neq);
  }

  // Calculate sensitivities
  std::vector<double> sensitivities = userData->sensitivities(t, states, sensCur, parameters);

  // Copy sensitivities into result container
  auto itcSens = sensitivities.cbegin();
  for(int i = 0; i < neq + npar; ++i) {
    std::copy(itcSens, itcSens + neq, NV_DATA_S(ySdot[i]));
    std::advance(itcSens, neq);
  }

  // Indicate success.
  return 0;
}

#endif
