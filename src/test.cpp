#define ARMA_DONT_USE_CXX11
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <array>
#include <vector>
using namespace std;
using namespace Rcpp;

/*
 This file contains a series of tests 
 */

extern "C" {
  array<vector<double>, 2> example_model_stl(const double& t, const vector<double>& states, 
            const vector<double>& parameters, const vector<double>& forcings) {
      
    vector<double> derivatives(states.size());  
    for(int i = 0; i < states.size(); i++) {
      derivatives[i] = -states[i]*parameters[0];
    }
    vector<double> observed{forcings[0]};
    array<vector<double>, 2> output{derivatives, observed};
    return output;
  }
  
  arma::mat example_jacobian_stl(const double& t, const vector<double>& states, 
            const vector<double>& parameters, const vector<double>& forcings) {
    arma::mat output = arma::eye(states.size(), states.size());
    output = -parameters[0]*output;
    return output;
  }
};
