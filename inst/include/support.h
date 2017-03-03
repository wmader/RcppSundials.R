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
