/*
 * DOTk_MexObjectiveFunction.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <cassert>

#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexObjectiveFunction.hpp"

namespace dotk
{

DOTk_MexObjectiveFunction::DOTk_MexObjectiveFunction(const mxArray* operators_, const dotk::types::problem_t & type_) :
        m_Value(nullptr),
        m_Gradient(nullptr),
        m_Hessian(nullptr),
        m_PartialDerivativeState(nullptr),
        m_PartialDerivativeControl(nullptr),
        m_PartialDerivativeStateState(nullptr),
        m_PartialDerivativeStateControl(nullptr),
        m_PartialDerivativeControlState(nullptr),
        m_PartialDerivativeControlControl(nullptr)
{
    this->initialize(operators_, type_);
}

DOTk_MexObjectiveFunction::~DOTk_MexObjectiveFunction()
{
    this->clear();
}

double DOTk_MexObjectiveFunction::value(const dotk::Vector<double> & control_)
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

void DOTk_MexObjectiveFunction::gradient(const dotk::Vector<double> & control_, dotk::Vector<double> & output_)
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

void DOTk_MexObjectiveFunction::hessian(const dotk::Vector<double> & control_,
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

double DOTk_MexObjectiveFunction::value(const dotk::Vector<double> & state_, const dotk::Vector<double> & control_)
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

void DOTk_MexObjectiveFunction::partialDerivativeState(const dotk::Vector<double> & state_,
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

void DOTk_MexObjectiveFunction::partialDerivativeControl(const dotk::Vector<double> & state_,
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

void DOTk_MexObjectiveFunction::partialDerivativeStateState(const dotk::Vector<double> & state_,
                                                            const dotk::Vector<double> & control_,
                                                            const dotk::Vector<double> & vector_,
                                                            dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_PartialDerivativeStateState, mx_state, mx_control, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeStateState.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexObjectiveFunction::partialDerivativeStateControl(const dotk::Vector<double> & state_,
                                                              const dotk::Vector<double> & control_,
                                                              const dotk::Vector<double> & vector_,
                                                              dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_PartialDerivativeStateControl, mx_state, mx_control, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeStateControl.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexObjectiveFunction::partialDerivativeControlState(const dotk::Vector<double> & state_,
                                                              const dotk::Vector<double> & control_,
                                                              const dotk::Vector<double> & vector_,
                                                              dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_PartialDerivativeControlState, mx_state, mx_control, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeControlState.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexObjectiveFunction::partialDerivativeControlControl(const dotk::Vector<double> & state_,
                                                                const dotk::Vector<double> & control_,
                                                                const dotk::Vector<double> & vector_,
                                                                dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_PartialDerivativeControlControl, mx_state, mx_control, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeControlControl.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexObjectiveFunction::clear()
{
    dotk::mex::destroy(m_Value);
    dotk::mex::destroy(m_Gradient);
    dotk::mex::destroy(m_Hessian);
    dotk::mex::destroy(m_PartialDerivativeState);
    dotk::mex::destroy(m_PartialDerivativeControl);
    dotk::mex::destroy(m_PartialDerivativeStateState);
    dotk::mex::destroy(m_PartialDerivativeStateControl);
    dotk::mex::destroy(m_PartialDerivativeControlState);
    dotk::mex::destroy(m_PartialDerivativeControlControl);
}

void DOTk_MexObjectiveFunction::initialize(const mxArray* operators_, const dotk::types::problem_t & type_)
{
    m_Value = mxDuplicateArray(mxGetField(operators_, 0, "value"));
    switch(type_)
    {
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_ILP:
        {
            m_Gradient = mxDuplicateArray(mxGetField(operators_, 0, "gradient"));
            m_Hessian = mxDuplicateArray(mxGetField(operators_, 0, "hessian"));
            break;
        }
        case dotk::types::TYPE_UNLP:
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_NLP_BOUND:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::TYPE_CNLP:
        {
            m_PartialDerivativeState = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeState"));
            m_PartialDerivativeControl = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControl"));
            m_PartialDerivativeStateState =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeStateState"));
            m_PartialDerivativeStateControl =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeStateControl"));
            m_PartialDerivativeControlState =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControlState"));
            m_PartialDerivativeControlControl =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControlControl"));
            break;
        }
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::ostringstream error;
            error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling initialize.\n";
            mexErrMsgTxt(error.str().c_str());
            break;
        }
    }
}

}
