/*
 * DOTk_MexFactoriesAlgorithmTypeGB.cpp
 *
 *  Created on: Apr 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexQuasiNewtonParser.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_TrustRegionMngTypeULP.hpp"
#include "DOTk_TrustRegionMngTypeUNP.hpp"
#include "DOTk_MexKrylovSolverParser.hpp"
#include "DOTk_MexFiniteDiffNumIntgParser.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"
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
    size_t itr = 0;
    dotk::mex::parseMaxNumKrylovSolverItr(options_, itr);
    dotk::types::krylov_solver_t type = dotk::types::KRYLOV_SOLVER_DISABLED;
    dotk::mex::parseKrylovSolverMethod(options_, type);

    switch(type)
    {
        case dotk::types::LEFT_PREC_CG:
        {
            algorithm_.setLeftPrecCgKrylovSolver(primal_, itr);
            break;
        }
        case dotk::types::PREC_GMRES:
        {
            algorithm_.setPrecGmresKrylovSolver(primal_, itr);
            break;
        }
        case dotk::types::LEFT_PREC_CGNR:
        {
            algorithm_.setLeftPrecCgnrKrylovSolver(primal_, itr);
            break;
        }
        case dotk::types::LEFT_PREC_CGNE:
        {
            algorithm_.setLeftPrecCgneKrylovSolver(primal_, itr);
            break;
        }
        case dotk::types::LEFT_PREC_CR:
        {
            algorithm_.setLeftPrecCrKrylovSolver(primal_, itr);
            break;
        }
        case dotk::types::LEFT_PREC_GCR:
        {
            algorithm_.setLeftPrecGcrKrylovSolver(primal_, itr);
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
            algorithm_.setLeftPrecCgKrylovSolver(primal_, itr);
            std::string msg(" DOTk/MEX WARNING: Invalid Krylov Solver Method. Default = CONUGATE GRADIENT. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
}

template<typename Manager>
void buildTrustRegionMethod(const mxArray* options_, Manager & mng_)
{
    dotk::types::trustregion_t type = dotk::types::TRUST_REGION_DISABLED;
    dotk::mex::parseTrustRegionMethod(options_, type);
    double initial_trust_region_radius = 1;
    dotk::mex::parseInitialTrustRegionRadius(options_, initial_trust_region_radius);

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
            mng_->setDoglegTrustRegionMethod(initial_trust_region_radius);
            std::string msg(" DOTk/MEX WARNING: Invalid Trust Region Method. Default = Dogleg. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
}

template<typename Manager>
void buildGradient(const mxArray* options_, const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_, Manager & mng_)
{
    dotk::types::gradient_t type = dotk::types::GRADIENT_OPERATOR_DISABLED;
    dotk::mex::parseGradientComputationMethod(options_, type);
    switch(type)
    {
        case dotk::types::FORWARD_DIFF_GRAD:
        {
            dotk::mex::parseFiniteDifferencePerturbation(options_, *epsilon_->control());
            mng_->setForwardFiniteDiffGradient(epsilon_);
            break;
        }
        case dotk::types::BACKWARD_DIFF_GRAD:
        {
            dotk::mex::parseFiniteDifferencePerturbation(options_, *epsilon_->control());
            mng_->setBackwardFiniteDiffGradient(epsilon_);
            break;
        }
        case dotk::types::CENTRAL_DIFF_GRAD:
        {
            dotk::mex::parseFiniteDifferencePerturbation(options_, *epsilon_->control());
            mng_->setCentralFiniteDiffGradient(epsilon_);
            break;
        }
        case dotk::types::USER_DEFINED_GRAD:
        {
            mng_->setUserDefinedGradient();
            break;
        }
        case dotk::types::PARALLEL_FORWARD_DIFF_GRAD:
        {
            dotk::mex::parseFiniteDifferencePerturbation(options_, *epsilon_->control());
            mng_->setParallelForwardFiniteDiffGradient(epsilon_);
            break;
        }
        case dotk::types::PARALLEL_BACKWARD_DIFF_GRAD:
        {
            dotk::mex::parseFiniteDifferencePerturbation(options_, *epsilon_->control());
            mng_->setParallelBackwardFiniteDiffGradient(epsilon_);
            break;
        }
        case dotk::types::PARALLEL_CENTRAL_DIFF_GRAD:
        {
            dotk::mex::parseFiniteDifferencePerturbation(options_, *epsilon_->control());
            mng_->setParallelCentralFiniteDiffGradient(epsilon_);
            break;
        }
        case dotk::types::GRADIENT_OPERATOR_DISABLED:
        {
            dotk::mex::parseFiniteDifferencePerturbation(options_, *epsilon_->control());
            mng_->setForwardFiniteDiffGradient(epsilon_);
            std::string msg(" DOTk/MEX WARNING: Invalid Gradient Method. Default = FORWARD DIFFERENCE. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
}

}

}
