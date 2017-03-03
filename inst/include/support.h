#include <vector>
#include <array>
#include <stdexcept>
#include <datatypes.h>

void checkModel(int t, int neq,
                const std::vector<double>& initials,
                const std::vector<double>& parameters,
                statesRHS* model) {
  std::array<std::vector<double>, 2> states;

  try {
    states = model(t, initials, parameters);
  } catch(std::exception &ex) {
    forward_exception_to_r(ex);
  } catch(...) {
    ::Rf_error("First call to model failed for unknown reasons.");
  }

  // Check that the number of time derivatives matches with settings
  if(states[0].size() != neq) {
  ::Rf_error("Mismatch between number of state initials and derivatives returned from model.");
  }
}



void* createCVodes(const Rcpp::List& settings) {
  void* cvode_mem = nullptr;

  if(Rcpp::as<std::string>(settings["method"]) == "bdf") {
    cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  } else if(Rcpp::as<std::string>(settings["method"]) == "adams"){
    cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
  } else {
    throw std::invalid_argument("Please choose bdf or adams as method");
  }

  if(cvode_mem == nullptr) std::runtime_error("Could not create the CVODES solver object.");
  return cvode_mem;
}



void cvSuccess(const int flag, const std::string& msg) {
  if(flag < CV_SUCCESS) {
    throw std::runtime_error(std::string("Initializing CVodes failed:\n") + msg);
  }
}
