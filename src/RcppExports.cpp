// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/RcppSundials.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <string>
#include <set>

using namespace Rcpp;

// cvode_R
NumericMatrix cvode_R(NumericVector times, NumericVector states, NumericVector parameters, List forcings_data, List settings, Function model, Function jacobian);
static SEXP RcppSundials_cvode_R_try(SEXP timesSEXP, SEXP statesSEXP, SEXP parametersSEXP, SEXP forcings_dataSEXP, SEXP settingsSEXP, SEXP modelSEXP, SEXP jacobianSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::traits::input_parameter< NumericVector >::type times(timesSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type states(statesSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type parameters(parametersSEXP );
        Rcpp::traits::input_parameter< List >::type forcings_data(forcings_dataSEXP );
        Rcpp::traits::input_parameter< List >::type settings(settingsSEXP );
        Rcpp::traits::input_parameter< Function >::type model(modelSEXP );
        Rcpp::traits::input_parameter< Function >::type jacobian(jacobianSEXP );
        NumericMatrix __result = cvode_R(times, states, parameters, forcings_data, settings, model, jacobian);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP_RETURN_ERROR
}
RcppExport SEXP RcppSundials_cvode_R(SEXP timesSEXP, SEXP statesSEXP, SEXP parametersSEXP, SEXP forcings_dataSEXP, SEXP settingsSEXP, SEXP modelSEXP, SEXP jacobianSEXP) {
    SEXP __result;
    {
        Rcpp::RNGScope __rngScope;
        __result = PROTECT(RcppSundials_cvode_R_try(timesSEXP, statesSEXP, parametersSEXP, forcings_dataSEXP, settingsSEXP, modelSEXP, jacobianSEXP));
    }
    Rboolean __isInterrupt = Rf_inherits(__result, "interrupted-error");
    if (__isInterrupt) {
        UNPROTECT(1);
        Rf_onintr();
    }
    Rboolean __isError = Rf_inherits(__result, "try-error");
    if (__isError) {
        SEXP __msgSEXP = Rf_asChar(__result);
        UNPROTECT(1);
        Rf_error(CHAR(__msgSEXP));
    }
    UNPROTECT(1);
    return __result;
}
// cvode_Cpp
NumericMatrix cvode_Cpp(NumericVector times, NumericVector states, NumericVector parameters, List forcings_data, List settings, SEXP model_, SEXP jacobian_);
static SEXP RcppSundials_cvode_Cpp_try(SEXP timesSEXP, SEXP statesSEXP, SEXP parametersSEXP, SEXP forcings_dataSEXP, SEXP settingsSEXP, SEXP model_SEXP, SEXP jacobian_SEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::traits::input_parameter< NumericVector >::type times(timesSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type states(statesSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type parameters(parametersSEXP );
        Rcpp::traits::input_parameter< List >::type forcings_data(forcings_dataSEXP );
        Rcpp::traits::input_parameter< List >::type settings(settingsSEXP );
        Rcpp::traits::input_parameter< SEXP >::type model_(model_SEXP );
        Rcpp::traits::input_parameter< SEXP >::type jacobian_(jacobian_SEXP );
        NumericMatrix __result = cvode_Cpp(times, states, parameters, forcings_data, settings, model_, jacobian_);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP_RETURN_ERROR
}
RcppExport SEXP RcppSundials_cvode_Cpp(SEXP timesSEXP, SEXP statesSEXP, SEXP parametersSEXP, SEXP forcings_dataSEXP, SEXP settingsSEXP, SEXP model_SEXP, SEXP jacobian_SEXP) {
    SEXP __result;
    {
        Rcpp::RNGScope __rngScope;
        __result = PROTECT(RcppSundials_cvode_Cpp_try(timesSEXP, statesSEXP, parametersSEXP, forcings_dataSEXP, settingsSEXP, model_SEXP, jacobian_SEXP));
    }
    Rboolean __isInterrupt = Rf_inherits(__result, "interrupted-error");
    if (__isInterrupt) {
        UNPROTECT(1);
        Rf_onintr();
    }
    Rboolean __isError = Rf_inherits(__result, "try-error");
    if (__isError) {
        SEXP __msgSEXP = Rf_asChar(__result);
        UNPROTECT(1);
        Rf_error(CHAR(__msgSEXP));
    }
    UNPROTECT(1);
    return __result;
}
// cvode_Cpp_stl
NumericMatrix cvode_Cpp_stl(NumericVector times, NumericVector states_, NumericVector parameters_, List forcings_data_, List settings, SEXP model_, SEXP jacobian_);
static SEXP RcppSundials_cvode_Cpp_stl_try(SEXP timesSEXP, SEXP states_SEXP, SEXP parameters_SEXP, SEXP forcings_data_SEXP, SEXP settingsSEXP, SEXP model_SEXP, SEXP jacobian_SEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::traits::input_parameter< NumericVector >::type times(timesSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type states_(states_SEXP );
        Rcpp::traits::input_parameter< NumericVector >::type parameters_(parameters_SEXP );
        Rcpp::traits::input_parameter< List >::type forcings_data_(forcings_data_SEXP );
        Rcpp::traits::input_parameter< List >::type settings(settingsSEXP );
        Rcpp::traits::input_parameter< SEXP >::type model_(model_SEXP );
        Rcpp::traits::input_parameter< SEXP >::type jacobian_(jacobian_SEXP );
        NumericMatrix __result = cvode_Cpp_stl(times, states_, parameters_, forcings_data_, settings, model_, jacobian_);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP_RETURN_ERROR
}
RcppExport SEXP RcppSundials_cvode_Cpp_stl(SEXP timesSEXP, SEXP states_SEXP, SEXP parameters_SEXP, SEXP forcings_data_SEXP, SEXP settingsSEXP, SEXP model_SEXP, SEXP jacobian_SEXP) {
    SEXP __result;
    {
        Rcpp::RNGScope __rngScope;
        __result = PROTECT(RcppSundials_cvode_Cpp_stl_try(timesSEXP, states_SEXP, parameters_SEXP, forcings_data_SEXP, settingsSEXP, model_SEXP, jacobian_SEXP));
    }
    Rboolean __isInterrupt = Rf_inherits(__result, "interrupted-error");
    if (__isInterrupt) {
        UNPROTECT(1);
        Rf_onintr();
    }
    Rboolean __isError = Rf_inherits(__result, "try-error");
    if (__isError) {
        SEXP __msgSEXP = Rf_asChar(__result);
        UNPROTECT(1);
        Rf_error(CHAR(__msgSEXP));
    }
    UNPROTECT(1);
    return __result;
}
// example_model
List example_model(double t, NumericVector states, NumericVector parameters, NumericVector forcings);
RcppExport SEXP RcppSundials_example_model(SEXP tSEXP, SEXP statesSEXP, SEXP parametersSEXP, SEXP forcingsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< double >::type t(tSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type states(statesSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type parameters(parametersSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type forcings(forcingsSEXP );
        List __result = example_model(t, states, parameters, forcings);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// example_jacobian
NumericMatrix example_jacobian(double t, NumericVector states, NumericVector parameters, NumericVector forcings);
RcppExport SEXP RcppSundials_example_jacobian(SEXP tSEXP, SEXP statesSEXP, SEXP parametersSEXP, SEXP forcingsSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< double >::type t(tSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type states(statesSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type parameters(parametersSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type forcings(forcingsSEXP );
        NumericMatrix __result = example_jacobian(t, states, parameters, forcings);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}

// validate (ensure exported C++ functions exist before calling them)
static int RcppSundials_RcppExport_validate(const char* sig) { 
    static std::set<std::string> signatures;
    if (signatures.empty()) {
        signatures.insert("NumericMatrix(*cvode_R)(NumericVector,NumericVector,NumericVector,List,List,Function,Function)");
        signatures.insert("NumericMatrix(*cvode_Cpp)(NumericVector,NumericVector,NumericVector,List,List,SEXP,SEXP)");
        signatures.insert("NumericMatrix(*cvode_Cpp_stl)(NumericVector,NumericVector,NumericVector,List,List,SEXP,SEXP)");
    }
    return signatures.find(sig) != signatures.end();
}

// registerCCallable (register entry points for exported C++ functions)
RcppExport SEXP RcppSundials_RcppExport_registerCCallable() { 
    R_RegisterCCallable("RcppSundials", "RcppSundials_cvode_R", (DL_FUNC)RcppSundials_cvode_R_try);
    R_RegisterCCallable("RcppSundials", "RcppSundials_cvode_Cpp", (DL_FUNC)RcppSundials_cvode_Cpp_try);
    R_RegisterCCallable("RcppSundials", "RcppSundials_cvode_Cpp_stl", (DL_FUNC)RcppSundials_cvode_Cpp_stl_try);
    R_RegisterCCallable("RcppSundials", "RcppSundials_RcppExport_validate", (DL_FUNC)RcppSundials_RcppExport_validate);
    return R_NilValue;
}
