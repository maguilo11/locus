/*
 * DOTk_MexNumDiffHessianFactory.cpp
 *
 *  Created on: Oct 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "vector.hpp"
#include "DOTk_MexNumDiffHessianFactory.hpp"
#include "DOTk_MexFiniteDiffNumIntgParser.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace dotk
{

namespace mex
{

void buildNumericallyDifferentiatedHessian(const mxArray* options_,
                                           const Vector<double> & input_,
                                           std::shared_ptr<dotk::NumericallyDifferentiatedHessian> & output_)
{
    double epsilon = dotk::mex::parseNumericalDifferentiationEpsilon(options_);
    dotk::types::numerical_integration_t type = dotk::mex::parseNumericalDifferentiationMethod(options_);
    switch(type)
    {
        case dotk::types::FORWARD_FINITE_DIFF:
        {
            output_->setForwardDifference(input_, epsilon);
            break;
        }
        case dotk::types::BACKWARD_FINITE_DIFF:
        {
            output_->setBackwardDifference(input_, epsilon);
            break;
        }
        case dotk::types::CENTRAL_FINITE_DIFF:
        {
            output_->setCentralDifference(input_, epsilon);
            break;
        }
        case dotk::types::SECOND_ORDER_FORWARD_FINITE_DIFF:
        {
            output_->setSecondOrderForwardDifference(input_, epsilon);
            break;
        }
        case dotk::types::THIRD_ORDER_FORWARD_FINITE_DIFF:
        {
            output_->setThirdOrderForwardDifference(input_, epsilon);
            break;
        }
        case dotk::types::THIRD_ORDER_BACKWARD_FINITE_DIFF:
        {
            output_->setThirdOrderBackwardDifference(input_, epsilon);
            break;
        }
        default:
        {
            std::ostringstream msg;
            msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> Numerical Differentiation Method keyword is misspelled."
                    << " Numerical Differentiation Method set to FORWARD FINITE DIFFERENCE.\n";
            mexWarnMsgTxt(msg.str().c_str());
            output_->setForwardDifference(input_, epsilon);
            break;
        }
    }
}

}

}
