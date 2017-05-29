#include <vector>
#include <array>
#include <stdexcept>
#include <datatypes.h>
#include <nvector/nvector_serial.h>

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



void cvSuccess(int flag, const std::string& msg) {
  if(flag < CV_SUCCESS) {
    throw std::runtime_error(std::string("Initializing CVodes failed:\n") + msg);
  }
}



void setEvent(void* cvode_mem, std::vector<Event>& events, N_Vector y, N_Vector* yS,
			  double currentTime, int neq) {
	Event event;

	while(events.size() > 0) {
		event = events.back();
		if (event.time == currentTime) {
			events.pop_back();
		} else {
			break;
		}

		if (event.variable < neq ) {
			// This is a state
			switch (event.method) {
				case replace :
					NV_Ith_S(y, event.variable) = event.value;
					break;
				case add :
					NV_Ith_S(y, event.variable) += event.value;
					break;
				case multiply :
					NV_Ith_S(y, event.variable) *= event.value;
					break;
				default : ::Rf_error("Failure: Unknown event type."); break;
			}
		} else {
			// This is a sensitivity
			int variable = event.variable - neq;
			int col = static_cast<int>(variable / neq);
			int row = variable % neq;

			switch (event.method) {
				case replace :
					NV_Ith_S(yS[col], row) = event.value;
					break;
				case add :
					::Rf_error("Failure: Event method add is not allowed to change sensitivities.");
					break;
				case multiply :
					NV_Ith_S(yS[col], row) *= event.value;
					break;
				default : ::Rf_error("Failure: Unknown event type."); break;
			}
		}
	}

	// Reset states and sensitivities
	int flag = CVodeReInit(cvode_mem, currentTime, y);
	cvSuccess(flag, "Failure: CVode could not be re-initialized.");
	flag = CVodeSensReInit(cvode_mem, CV_SIMULTANEOUS, yS);
	cvSuccess(flag, "Failure: Sensitivities could not be re-initialized.");
}



void storeResult(void* cvode_mem, N_Vector y, N_Vector* yS,
				 arma::mat& outputStates, arma::mat& outputSensitivities,
				 double tretSensitivities, int t, int neq, int Ns) {
	// Copy states to output container
	std::copy(NV_DATA_S(y), NV_DATA_S(y) + neq, outputStates.begin_col(t));

	// Copy sensitivities to output container
	int flag = CVodeGetSens(cvode_mem, &tretSensitivities, yS);
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
