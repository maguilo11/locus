/*
 * DOTk_MexEqualityConstraint.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <cassert>
#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexEqualityConstraint.hpp"

namespace dotk
{

DOTk_MexEqualityConstraint::DOTk_MexEqualityConstraint(const mxArray* operators_, const dotk::types::problem_t & type_) :
        m_Solve(nullptr),
        m_Residual(nullptr),
        m_ApplyInverseJacobianState(nullptr),
        m_ApplyAdjointInverseJacobianState(nullptr),
        m_Jacobian(nullptr),
        m_AdjointJacobian(nullptr),
        m_PartialDerivativeState(nullptr),
        m_PartialDerivativeControl(nullptr),
        m_AdjointPartialDerivativeState(nullptr),
        m_AdjointPartialDerivativeControl(nullptr),
        m_Hessian(nullptr),
        m_PartialDerivativeStateState(nullptr),
        m_PartialDerivativeStateControl(nullptr),
        m_PartialDerivativeControlState(nullptr),
        m_PartialDerivativeControlControl(nullptr)
{
    this->initialize(operators_, type_);
}

DOTk_MexEqualityConstraint::~DOTk_MexEqualityConstraint()
{
    this->clear();
}

void DOTk_MexEqualityConstraint::solve(const dotk::Vector<double> & control_, dotk::Vector<double> & output_)
{
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    mxArray* mx_output[1];
    mxArray* mx_input[2] = { m_Solve, mx_control };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 2, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling solve.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::applyInverseJacobianState(const dotk::Vector<double> & state_,
                                                           const dotk::Vector<double> & control_,
                                                           const dotk::Vector<double> & rhs_,
                                                           dotk::Vector<double> & output_)
{
    const dotk::MexVector & rhs = dynamic_cast<const dotk::MexVector &>(rhs_);
    mxArray* mx_rhs = const_cast<mxArray*>(rhs.array());
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_ApplyInverseJacobianState, mx_state, mx_control, mx_rhs };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling applyInverseJacobianState.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::applyAdjointInverseJacobianState(const dotk::Vector<double> & state_,
                                                                  const dotk::Vector<double> & control_,
                                                                  const dotk::Vector<double> & rhs_,
                                                                  dotk::Vector<double> & output_)
{
    const dotk::MexVector & rhs = dynamic_cast<const dotk::MexVector &>(rhs_);
    mxArray* mx_rhs = const_cast<mxArray*>(rhs.array());
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_ApplyAdjointInverseJacobianState, mx_state, mx_control, mx_rhs };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling applyAdjointInverseJacobianState.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::residual(const dotk::Vector<double> & primal_, dotk::Vector<double> & output_)
{
    const dotk::MexVector & primal = dynamic_cast<const dotk::MexVector &>(primal_);
    mxArray* mx_primal = const_cast<mxArray*>(primal.array());

    mxArray* mx_output[1];
    mxArray* mx_input[2] = { m_Residual, mx_primal };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 2, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling residual.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::jacobian(const dotk::Vector<double> & primal_,
                                          const dotk::Vector<double> & vector_,
                                          dotk::Vector<double> & output_)
{
    const dotk::MexVector & primal = dynamic_cast<const dotk::MexVector &>(primal_);
    mxArray* mx_primal = const_cast<mxArray*>(primal.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[3] = { m_Jacobian, mx_primal, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling jacobian.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::adjointJacobian(const dotk::Vector<double> & primal_,
                                                 const dotk::Vector<double> & dual_,
                                                 dotk::Vector<double> & output_)
{
    const dotk::MexVector & primal = dynamic_cast<const dotk::MexVector &>(primal_);
    mxArray* mx_primal = const_cast<mxArray*>(primal.array());
    const dotk::MexVector & dual = dynamic_cast<const dotk::MexVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());

    mxArray* mx_output[1];
    mxArray* mx_input[3] = { m_AdjointJacobian, mx_primal, mx_dual };

    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling adjointJacobian.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::hessian(const dotk::Vector<double> & primal_,
                                         const dotk::Vector<double> & dual_,
                                         const dotk::Vector<double> & vector_,
                                         dotk::Vector<double> & output_)
{
    const dotk::MexVector & primal = dynamic_cast<const dotk::MexVector &>(primal_);
    mxArray* mx_primal = const_cast<mxArray*>(primal.array());
    const dotk::MexVector & dual = dynamic_cast<const dotk::MexVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_Hessian, mx_primal, mx_dual, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling hessian.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::residual(const dotk::Vector<double> & state_,
                                          const dotk::Vector<double> & control_,
                                          dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());

    mxArray* mx_output[1];
    mxArray* mx_input[3] = { m_Residual, mx_state, mx_control };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling residual.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::partialDerivativeState(const dotk::Vector<double> & state_,
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
    mxArray* mx_input[4] = { m_PartialDerivativeState, mx_state, mx_control, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeState.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::partialDerivativeControl(const dotk::Vector<double> & state_,
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
    mxArray* mx_input[4] = { m_PartialDerivativeControl, mx_state, mx_control, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling partialDerivativeControl.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::adjointPartialDerivativeState(const dotk::Vector<double> & state_,
                                                               const dotk::Vector<double> & control_,
                                                               const dotk::Vector<double> & dual_,
                                                               dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & dual = dynamic_cast<const dotk::MexVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());

    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_AdjointPartialDerivativeState, mx_state, mx_control, mx_dual };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling adjointPartialDerivativeState.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::adjointPartialDerivativeControl(const dotk::Vector<double> & state_,
                                                                 const dotk::Vector<double> & control_,
                                                                 const dotk::Vector<double> & dual_,
                                                                 dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & dual = dynamic_cast<const dotk::MexVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());

    mxArray* mx_output[1];
    mxArray* mx_input[4] = { m_AdjointPartialDerivativeControl, mx_state, mx_control, mx_dual };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling adjointPartialDerivativeControl.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::partialDerivativeStateState(const dotk::Vector<double> & state_,
                                                             const dotk::Vector<double> & control_,
                                                             const dotk::Vector<double> & dual_,
                                                             const dotk::Vector<double> & vector_,
                                                             dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & dual = dynamic_cast<const dotk::MexVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[5] = { m_PartialDerivativeStateState, mx_state, mx_control, mx_dual, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling partialDerivativeStateState.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::partialDerivativeStateControl(const dotk::Vector<double> & state_,
                                                               const dotk::Vector<double> & control_,
                                                               const dotk::Vector<double> & dual_,
                                                               const dotk::Vector<double> & vector_,
                                                               dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & dual = dynamic_cast<const dotk::MexVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[5] = { m_PartialDerivativeStateControl, mx_state, mx_control, mx_dual, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling partialDerivativeStateControl.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::partialDerivativeControlControl(const dotk::Vector<double> & state_,
                                                                 const dotk::Vector<double> & control_,
                                                                 const dotk::Vector<double> & dual_,
                                                                 const dotk::Vector<double> & vector_,
                                                                 dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & dual = dynamic_cast<const dotk::MexVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[5] = { m_PartialDerivativeControlControl, mx_state, mx_control, mx_dual, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling partialDerivativeControlControl.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::partialDerivativeControlState(const dotk::Vector<double> & state_,
                                                               const dotk::Vector<double> & control_,
                                                               const dotk::Vector<double> & dual_,
                                                               const dotk::Vector<double> & vector_,
                                                               dotk::Vector<double> & output_)
{
    const dotk::MexVector & state = dynamic_cast<const dotk::MexVector &>(state_);
    mxArray* mx_state = const_cast<mxArray*>(state.array());
    const dotk::MexVector & control = dynamic_cast<const dotk::MexVector &>(control_);
    mxArray* mx_control = const_cast<mxArray*>(control.array());
    const dotk::MexVector & dual = dynamic_cast<const dotk::MexVector &>(dual_);
    mxArray* mx_dual = const_cast<mxArray*>(dual.array());
    const dotk::MexVector & vector = dynamic_cast<const dotk::MexVector &>(vector_);
    mxArray* mx_vector = const_cast<mxArray*>(vector.array());

    mxArray* mx_output[1];
    mxArray* mx_input[5] = { m_PartialDerivativeControlState, mx_state, mx_control, mx_dual, mx_vector };
    mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
    std::ostringstream msg;
    msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
            << ", -> Error while calling partialDerivativeControlState.\n";
    dotk::mex::handleException(error, msg.str());

    assert(output_.size() == mxGetNumberOfElements(mx_output[0]));
    dotk::MexVector & output = dynamic_cast<dotk::MexVector &>(output_);
    output.setMxArray(mx_output[0]);
}

void DOTk_MexEqualityConstraint::clear()
{
    dotk::mex::destroy(m_Solve);
    dotk::mex::destroy(m_Residual);
    dotk::mex::destroy(m_ApplyInverseJacobianState);
    dotk::mex::destroy(m_ApplyAdjointInverseJacobianState);

    dotk::mex::destroy(m_Jacobian);
    dotk::mex::destroy(m_AdjointJacobian);
    dotk::mex::destroy(m_PartialDerivativeState);
    dotk::mex::destroy(m_PartialDerivativeControl);
    dotk::mex::destroy(m_AdjointPartialDerivativeState);
    dotk::mex::destroy(m_AdjointPartialDerivativeControl);

    dotk::mex::destroy(m_Hessian);
    dotk::mex::destroy(m_PartialDerivativeStateState);
    dotk::mex::destroy(m_PartialDerivativeStateControl);
    dotk::mex::destroy(m_PartialDerivativeControlState);
    dotk::mex::destroy(m_PartialDerivativeControlControl);
}

void DOTk_MexEqualityConstraint::initialize(const mxArray* operators_, const dotk::types::problem_t & type_)
{
    m_Solve = mxDuplicateArray(mxGetField(operators_, 0, "solve"));
    m_Residual = mxDuplicateArray(mxGetField(operators_, 0, "residual"));
    switch(type_)
    {
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_ILP:
        {
            m_Jacobian = mxDuplicateArray(mxGetField(operators_, 0, "jacobian"));
            m_Hessian = mxDuplicateArray(mxGetField(operators_, 0, "hessian"));
            m_AdjointJacobian = mxDuplicateArray(mxGetField(operators_, 0, "adjointJacobian"));
            break;
        }
        case dotk::types::TYPE_UNLP:
        case dotk::types::TYPE_NLP_BOUND:
        {
            m_PartialDerivativeState = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeState"));
            m_PartialDerivativeControl = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControl"));
            m_ApplyInverseJacobianState = mxDuplicateArray(mxGetField(operators_, 0, "applyInverseJacobianState"));
            m_PartialDerivativeStateState = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeStateState"));
            m_AdjointPartialDerivativeState =
                    mxDuplicateArray(mxGetField(operators_, 0, "adjointPartialDerivativeState"));
            m_PartialDerivativeStateControl =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeStateControl"));
            m_PartialDerivativeControlState =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControlState"));
            m_AdjointPartialDerivativeControl =
                    mxDuplicateArray(mxGetField(operators_, 0, "adjointPartialDerivativeControl"));
            m_PartialDerivativeControlControl =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControlControl"));
            m_ApplyAdjointInverseJacobianState =
                    mxDuplicateArray(mxGetField(operators_, 0, "applyAdjointInverseJacobianState"));
            break;
        }
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::TYPE_CNLP:
        {
            m_PartialDerivativeState = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeState"));
            m_PartialDerivativeControl = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControl"));
            m_ApplyInverseJacobianState = mxDuplicateArray(mxGetField(operators_, 0, "applyInverseJacobianState"));
            m_PartialDerivativeStateState = mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeStateState"));
            m_AdjointPartialDerivativeState =
                    mxDuplicateArray(mxGetField(operators_, 0, "adjointPartialDerivativeState"));
            m_PartialDerivativeStateControl =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeStateControl"));
            m_PartialDerivativeControlState =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControlState"));
            m_AdjointPartialDerivativeControl =
                    mxDuplicateArray(mxGetField(operators_, 0, "adjointPartialDerivativeControl"));
            m_PartialDerivativeControlControl =
                    mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControlControl"));
            m_ApplyAdjointInverseJacobianState =
                    mxDuplicateArray(mxGetField(operators_, 0, "applyAdjointInverseJacobianState"));
            break;
        }
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::string error("\nERROR: Invalid Problem ScalarType in Call to DOTk_MexEqualityConstraint::initialize\n");
            mexErrMsgTxt(error.c_str());
            break;
        }
    }
}

}
