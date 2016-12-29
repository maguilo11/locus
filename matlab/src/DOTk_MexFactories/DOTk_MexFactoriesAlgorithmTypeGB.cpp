/*
 * DOTk_MexFactoriesAlgorithmTypeGB.cpp
 *
 *  Created on: Apr 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "DOTk_MexVector.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexQuasiNewtonParser.hpp"
#include "DOTk_MexKrylovSolverParser.hpp"
#include "DOTk_MexFiniteDiffNumIntgParser.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_TrustRegionMngTypeULP.hpp"
#include "DOTk_TrustRegionMngTypeUNP.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace dotk
{

namespace mex
{

template<typename Algorithm>
void buildKrylovSolver(const mxArray* options_,
                       const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                       Algorithm & algorithm_)
{
    size_t iterations = dotk::mex::parseMaxNumKrylovSolverItr(options_);
    dotk::types::krylov_solver_t type = dotk::mex::parseKrylovSolverMethod(options_);
    switch(type)
    {
        case dotk::types::LEFT_PREC_CG:
        {
            algorithm_.setLeftPrecCgKrylovSolver(primal_, iterations);
            break;
        }
        case dotk::types::PREC_GMRES:
        {
            algorithm_.setPrecGmresKrylovSolver(primal_, iterations);
            break;
        }
        case dotk::types::LEFT_PREC_CGNR:
        {
            algorithm_.setLeftPrecCgnrKrylovSolver(primal_, iterations);
            break;
        }
        case dotk::types::LEFT_PREC_CGNE:
        {
            algorithm_.setLeftPrecCgneKrylovSolver(primal_, iterations);
            break;
        }
        case dotk::types::LEFT_PREC_CR:
        {
            algorithm_.setLeftPrecCrKrylovSolver(primal_, iterations);
            break;
        }
        case dotk::types::LEFT_PREC_GCR:
        {
            algorithm_.setLeftPrecGcrKrylovSolver(primal_, iterations);
            break;
        }
        case dotk::types::PROJECTED_PREC_CG:
        case dotk::types::LANCZOS:
        case dotk::types::BICG:
        case dotk::types::BICG_STAB:
        case dotk::types::USER_DEFINED_KRYLOV_SOLVER:
        case dotk::types::KRYLOV_SOLVER_DISABLED:
        default:
        {
            std::ostringstream msg;
            msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> Krylov solver set to default type = PCG.\n";
            mexWarnMsgTxt(msg.str().c_str());
            algorithm_.setLeftPrecCgKrylovSolver(primal_, iterations);
            break;
        }
    }
}

template<typename Manager>
void buildTrustRegionMethod(const mxArray* options_, Manager & mng_)
{
    dotk::types::trustregion_t type = dotk::mex::parseTrustRegionMethod(options_);
    double initial_trust_region_radius = dotk::mex::parseInitialTrustRegionRadius(options_);
    switch(type)
    {
        case dotk::types::TRUST_REGION_DOGLEG:
        {
            mng_->setDoglegTrustRegionMethod(initial_trust_region_radius);
            break;
        }
        case dotk::types::TRUST_REGION_CAUCHY:
        {
            mng_->setCauchyTrustRegionMethod(initial_trust_region_radius);
            break;
        }
        case dotk::types::TRUST_REGION_DOUBLE_DOGLEG:
        {
            mng_->setDoubleDoglegTrustRegionMethod(mng_->getTrialStep(), initial_trust_region_radius);
            break;
        }
        default:
        {
            std::ostringstream msg;
            msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> Trust region method set to default type = DOGLEG.\n";
            mexWarnMsgTxt(msg.str().c_str());
            mng_->setDoglegTrustRegionMethod(initial_trust_region_radius);
            break;
        }
    }
}

template<typename Manager>
void buildGradient(const mxArray* options_, Manager & mng_)
{
    dotk::types::gradient_t type = dotk::mex::parseGradientComputationMethod(options_);
    switch(type)
    {
        case dotk::types::FORWARD_DIFF_GRAD:
        {
            mxArray* mx_epsilon = dotk::mex::parseFiniteDifferencePerturbation(options_);
            dotk::MexVector epsilon(mx_epsilon);
            mxDestroyArray(mx_epsilon);
            mng_->setForwardFiniteDiffGradient(epsilon);
            break;
        }
        case dotk::types::BACKWARD_DIFF_GRAD:
        {
            mxArray* mx_epsilon = dotk::mex::parseFiniteDifferencePerturbation(options_);
            dotk::MexVector epsilon(mx_epsilon);
            mxDestroyArray(mx_epsilon);
            mng_->setBackwardFiniteDiffGradient(epsilon);
            break;
        }
        case dotk::types::CENTRAL_DIFF_GRAD:
        {
            mxArray* mx_epsilon = dotk::mex::parseFiniteDifferencePerturbation(options_);
            dotk::MexVector epsilon(mx_epsilon);
            mxDestroyArray(mx_epsilon);
            mng_->setCentralFiniteDiffGradient(epsilon);
            break;
        }
        case dotk::types::USER_DEFINED_GRAD:
        {
            mng_->setUserDefinedGradient();
            break;
        }
        case dotk::types::PARALLEL_FORWARD_DIFF_GRAD:
        {
            mxArray* mx_epsilon = dotk::mex::parseFiniteDifferencePerturbation(options_);
            dotk::MexVector epsilon(mx_epsilon);
            mxDestroyArray(mx_epsilon);
            mng_->setParallelForwardFiniteDiffGradient(epsilon);
            break;
        }
        case dotk::types::PARALLEL_BACKWARD_DIFF_GRAD:
        {
            mxArray* mx_epsilon = dotk::mex::parseFiniteDifferencePerturbation(options_);
            dotk::MexVector epsilon(mx_epsilon);
            mxDestroyArray(mx_epsilon);
            mng_->setParallelBackwardFiniteDiffGradient(epsilon);
            break;
        }
        case dotk::types::PARALLEL_CENTRAL_DIFF_GRAD:
        {
            mxArray* mx_epsilon = dotk::mex::parseFiniteDifferencePerturbation(options_);
            dotk::MexVector epsilon(mx_epsilon);
            mxDestroyArray(mx_epsilon);
            mng_->setParallelCentralFiniteDiffGradient(epsilon);
            break;
        }
        case dotk::types::GRADIENT_OPERATOR_DISABLED:
        default:
        {
            std::ostringstream msg;
            msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> DEFAULT Gradient method set to Forward Difference.\n";
            mexWarnMsgTxt(msg.str().c_str());
            mxArray* mx_epsilon = dotk::mex::parseFiniteDifferencePerturbation(options_);
            dotk::MexVector epsilon(mx_epsilon);
            mxDestroyArray(mx_epsilon);
            mng_->setForwardFiniteDiffGradient(epsilon);
            break;
        }
    }
}

}

}
