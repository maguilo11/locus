/*
 * DOTk_MexNumDiffHessianFactory.cpp
 *
 *  Created on: Oct 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_MexNumDiffHessianFactory.hpp"
#include "DOTk_MexFiniteDiffNumIntgParser.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace dotk
{

namespace mex
{

void buildNumericallyDifferentiatedHessian(const mxArray* options_,
                                           const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                           std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> & hessian_)
{
    double epsilon = 0.;
    dotk::mex::parseNumericalDifferentiationEpsilon(options_, epsilon);
    dotk::types::numerical_integration_t type = dotk::types::NUM_INTG_DISABLED;
    dotk::mex::parseNumericalDifferentiationMethod(options_, type);

    switch(type)
    {
        case dotk::types::FORWARD_FINITE_DIFF:
        {
            hessian_->setForwardDifference(primal_, epsilon);
            break;
        }
        case dotk::types::BACKWARD_FINITE_DIFF:
        {
            hessian_->setBackwardDifference(primal_, epsilon);
            break;
        }
        case dotk::types::CENTRAL_FINITE_DIFF:
        {
            hessian_->setCentralDifference(primal_, epsilon);
            break;
        }
        case dotk::types::SECOND_ORDER_FORWARD_FINITE_DIFF:
        {
            hessian_->setSecondOrderForwardDifference(primal_, epsilon);
            break;
        }
        case dotk::types::THIRD_ORDER_FORWARD_FINITE_DIFF:
        {
            hessian_->setThirdOrderForwardDifference(primal_, epsilon);
            break;
        }
        case dotk::types::THIRD_ORDER_BACKWARD_FINITE_DIFF:
        {
            hessian_->setThirdOrderBackwardDifference(primal_, epsilon);
            break;
        }
        default:
        {
            hessian_->setForwardDifference(primal_, epsilon);
            std::string msg(" DOTk/MEX WARNING: Invalid Numerical Differentiation Method. Default = FORWARD DIFFERENCE. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
}

}

}
