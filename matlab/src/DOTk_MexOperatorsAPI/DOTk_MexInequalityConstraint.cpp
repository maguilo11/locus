/*
 * DOTk_MexInequalityConstraint.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <cassert>
#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexInequalityConstraint.hpp"

namespace dotk
{

DOTk_MexInequalityConstraint::DOTk_MexInequalityConstraint(const mxArray* operators_,
                                                           const dotk::types::problem_t & type_) :
        dotk::DOTk_InequalityConstraint<double>(),
        m_Value(nullptr),
        m_Bound(nullptr),
        m_Gradient(nullptr),
        m_Hessian(nullptr),
        m_PartialDerivativeState(nullptr),
        m_PartialDerivativeControl(nullptr)
{
    this->initialize(operators_, type_);
}

DOTk_MexInequalityConstraint::~DOTk_MexInequalityConstraint()
{
    this->clear();
}

double DOTk_MexInequalityConstraint::bound()
{
    mxArray* mx_output[1];
    mxArray* mx_input[1] = { m_Bound };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 1, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling bound.\n";
    dotk::mex::handleException(error, msg.str());

    assert(1u == mxGetNumberOfElements(mx_output[0]));
    double output = mxGetScalar(mx_output[0]);
    return (output);
}

double DOTk_MexInequalityConstraint::value(const dotk::Vector<double> & control_)
{
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    mxArray* mx_output[1];
    mxArray* mx_input[2] = { m_Value, mx_control };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 2, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling value.\n";
    dotk::mex::handleException(error, msg.str());

    assert(1u == mxGetNumberOfElements(mx_output[0]));
    double output = mxGetScalar(mx_output[0]);
    return (output);
}

void DOTk_MexInequalityConstraint::gradient(const dotk::Vector<double> & control_, dotk::Vector<double> & output_)
{
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    mxArray* mx_output[1];
    mxArray* mx_input[2] = { m_Gradient, mx_control };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 2, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling gradient.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexInequalityConstraint::hessian(const dotk::Vector<double> & control_,
                                           const dotk::Vector<double> & vector_,
                                           dotk::Vector<double> & output_)
{
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[3] = { m_Hessian, mx_control, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling hessian.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

double DOTk_MexInequalityConstraint::value(const dotk::Vector<double> & state_, const dotk::Vector<double> & control_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    mxArray* mx_output[1];
    mxArray* mx_input[3] = { m_Value, mx_state, mx_control };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling value.\n";
    dotk::mex::handleException(error, msg.str());

    assert(1u == mxGetNumberOfElements(mx_output[0]));
    double output = mxGetScalar(mx_output[0]);
    return (output);
}

void DOTk_MexInequalityConstraint::partialDerivativeState(const dotk::Vector<double> & state_,
                                                          const dotk::Vector<double> & control_,
                                                          dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    mxArray* mx_output[1];
    mxArray* mx_input[3] = { m_PartialDerivativeState, mx_state, mx_control };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeState.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexInequalityConstraint::partialDerivativeControl(const dotk::Vector<double> & state_,
                                                            const dotk::Vector<double> & control_,
                                                            dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    mxArray* mx_output[1];
    mxArray* mx_input[3] = { m_PartialDerivativeControl, mx_state, mx_control };

    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeControl.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexInequalityConstraint::clear()
{
    dotk::mex::destroy(m_Value);
    dotk::mex::destroy(m_Bound);
    dotk::mex::destroy(m_Gradient);
    dotk::mex::destroy(m_Hessian);
    dotk::mex::destroy(m_PartialDerivativeState);
    dotk::mex::destroy(m_PartialDerivativeControl);
}

void DOTk_MexInequalityConstraint::initialize(const mxArray* operators_, const dotk::types::problem_t & type_)
{
    m_Value = mxDuplicateArray(mxGetField(operators_, 0, "value"));
    m_Bound = mxDuplicateArray(mxGetField(operators_, 0, "bound"));

    switch(type_)
    {
        case dotk::types::TYPE_ILP:
        case dotk::types::TYPE_CLP:
        {
            m_Gradient = mxDuplicateArray(mxGetField(operators_, 0, "gradient"));
            break;
        }
        case dotk::types::TYPE_CNLP:
        {
            m_PartialDerivativeState = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeState"));
            m_PartialDerivativeControl = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControl"));
            break;
        }
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_UNLP:
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_NLP_BOUND:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::ostringstream msg;
            msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling initialize.\n";
            mexErrMsgTxt(msg.str().c_str());
            break;
        }
    }
}

}
