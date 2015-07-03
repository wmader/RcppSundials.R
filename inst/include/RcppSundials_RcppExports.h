// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#ifndef __RcppSundials_RcppExports_h__
#define __RcppSundials_RcppExports_h__

#include <RcppArmadillo.h>
#include <Rcpp.h>

namespace RcppSundials {

    using namespace Rcpp;

    namespace {
        void validateSignature(const char* sig) {
            Rcpp::Function require = Rcpp::Environment::base_env()["require"];
            require("RcppSundials", Rcpp::Named("quietly") = true);
            typedef int(*Ptr_validate)(const char*);
            static Ptr_validate p_validate = (Ptr_validate)
                R_GetCCallable("RcppSundials", "RcppSundials_RcppExport_validate");
            if (!p_validate(sig)) {
                throw Rcpp::function_not_exported(
                    "C++ function with signature '" + std::string(sig) + "' not found in RcppSundials");
            }
        }
    }

    inline NumericMatrix cvode_Cpp_stl(NumericVector times, NumericVector states_, NumericVector parameters_, List forcings_data_, List settings, SEXP model_, SEXP jacobian_) {
        typedef SEXP(*Ptr_cvode_Cpp_stl)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_cvode_Cpp_stl p_cvode_Cpp_stl = NULL;
        if (p_cvode_Cpp_stl == NULL) {
            validateSignature("NumericMatrix(*cvode_Cpp_stl)(NumericVector,NumericVector,NumericVector,List,List,SEXP,SEXP)");
            p_cvode_Cpp_stl = (Ptr_cvode_Cpp_stl)R_GetCCallable("RcppSundials", "RcppSundials_cvode_Cpp_stl");
        }
        RObject __result;
        {
            RNGScope __rngScope;
            __result = p_cvode_Cpp_stl(Rcpp::wrap(times), Rcpp::wrap(states_), Rcpp::wrap(parameters_), Rcpp::wrap(forcings_data_), Rcpp::wrap(settings), Rcpp::wrap(model_), Rcpp::wrap(jacobian_));
        }
        if (__result.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (__result.inherits("try-error"))
            throw Rcpp::exception(as<std::string>(__result).c_str());
        return Rcpp::as<NumericMatrix >(__result);
    }

    inline List cvode_calc_derivs(SEXP model_, NumericVector t, NumericVector states, NumericVector parameters, List forcings_data_) {
        typedef SEXP(*Ptr_cvode_calc_derivs)(SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_cvode_calc_derivs p_cvode_calc_derivs = NULL;
        if (p_cvode_calc_derivs == NULL) {
            validateSignature("List(*cvode_calc_derivs)(SEXP,NumericVector,NumericVector,NumericVector,List)");
            p_cvode_calc_derivs = (Ptr_cvode_calc_derivs)R_GetCCallable("RcppSundials", "RcppSundials_cvode_calc_derivs");
        }
        RObject __result;
        {
            RNGScope __rngScope;
            __result = p_cvode_calc_derivs(Rcpp::wrap(model_), Rcpp::wrap(t), Rcpp::wrap(states), Rcpp::wrap(parameters), Rcpp::wrap(forcings_data_));
        }
        if (__result.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (__result.inherits("try-error"))
            throw Rcpp::exception(as<std::string>(__result).c_str());
        return Rcpp::as<List >(__result);
    }

    inline NumericMatrix cvode_calc_jac(SEXP jacobian_, NumericVector t, NumericVector states, NumericVector parameters, List forcings_data_) {
        typedef SEXP(*Ptr_cvode_calc_jac)(SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_cvode_calc_jac p_cvode_calc_jac = NULL;
        if (p_cvode_calc_jac == NULL) {
            validateSignature("NumericMatrix(*cvode_calc_jac)(SEXP,NumericVector,NumericVector,NumericVector,List)");
            p_cvode_calc_jac = (Ptr_cvode_calc_jac)R_GetCCallable("RcppSundials", "RcppSundials_cvode_calc_jac");
        }
        RObject __result;
        {
            RNGScope __rngScope;
            __result = p_cvode_calc_jac(Rcpp::wrap(jacobian_), Rcpp::wrap(t), Rcpp::wrap(states), Rcpp::wrap(parameters), Rcpp::wrap(forcings_data_));
        }
        if (__result.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (__result.inherits("try-error"))
            throw Rcpp::exception(as<std::string>(__result).c_str());
        return Rcpp::as<NumericMatrix >(__result);
    }

    inline NumericMatrix cvode_R(NumericVector times, NumericVector states, NumericVector parameters, List forcings_data, List settings, Function model, Function jacobian) {
        typedef SEXP(*Ptr_cvode_R)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_cvode_R p_cvode_R = NULL;
        if (p_cvode_R == NULL) {
            validateSignature("NumericMatrix(*cvode_R)(NumericVector,NumericVector,NumericVector,List,List,Function,Function)");
            p_cvode_R = (Ptr_cvode_R)R_GetCCallable("RcppSundials", "RcppSundials_cvode_R");
        }
        RObject __result;
        {
            RNGScope __rngScope;
            __result = p_cvode_R(Rcpp::wrap(times), Rcpp::wrap(states), Rcpp::wrap(parameters), Rcpp::wrap(forcings_data), Rcpp::wrap(settings), Rcpp::wrap(model), Rcpp::wrap(jacobian));
        }
        if (__result.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (__result.inherits("try-error"))
            throw Rcpp::exception(as<std::string>(__result).c_str());
        return Rcpp::as<NumericMatrix >(__result);
    }

}

#endif // __RcppSundials_RcppExports_h__
