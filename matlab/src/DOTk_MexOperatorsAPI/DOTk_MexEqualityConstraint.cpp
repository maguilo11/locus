/*
 * DOTk_MexEqualityConstraint.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexEqualityConstraint.hpp"

namespace dotk
{

template<class Type>
DOTk_MexEqualityConstraint<Type>::DOTk_MexEqualityConstraint(const mxArray* operators_,
                                                             const dotk::types::problem_t & type_) :
        m_Solve(NULL),
        m_Hessian(NULL),
        m_Residual(NULL),
        m_Jacobian(NULL),
        m_PartialDerivativeState(NULL),
        m_PartialDerivativeControl(NULL),
        m_AdjointPartialDerivative(NULL),
        m_ApplyInverseJacobianState(NULL),
        m_PartialDerivativeStateState(NULL),
        m_AdjointPartialDerivativeState(NULL),
        m_PartialDerivativeStateControl(NULL),
        m_PartialDerivativeControlState(NULL),
        m_AdjointPartialDerivativeControl(NULL),
        m_PartialDerivativeControlControl(NULL),
        m_ApplyAdjointInverseJacobianState(NULL)
{
    this->initialize(operators_, type_);
}

template<class Type>
DOTk_MexEqualityConstraint<Type>::~DOTk_MexEqualityConstraint()
{
    this->clear();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::solve(const dotk::vector<Type> & control_, dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[2] =
        { m_Solve.get(), control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 2, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::solve");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    control.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::applyInverseJacobianState(const dotk::vector<Type> & state_,
                                                                 const dotk::vector<Type> & control_,
                                                                 const dotk::vector<Type> & rhs_,
                                                                 dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr rhs(mxCreateDoubleMatrix(rhs_.size(), 1, mxREAL));
    rhs_.gather(mxGetPr(rhs.get()));

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[4] =
        { m_ApplyInverseJacobianState.get(), state.get(), control.get(), rhs.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::applyInverseJacobianState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    rhs.release();
    state.release();
    control.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::applyAdjointInverseJacobianState(const dotk::vector<Type> & state_,
                                                                        const dotk::vector<Type> & control_,
                                                                        const dotk::vector<Type> & rhs_,
                                                                        dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr rhs(mxCreateDoubleMatrix(rhs_.size(), 1, mxREAL));
    rhs_.gather(mxGetPr(rhs.get()));

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[4] =
        { m_ApplyAdjointInverseJacobianState.get(), state.get(), control.get(), rhs.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::applyAdjointInverseJacobianState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    rhs.release();
    state.release();
    control.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::residual(const dotk::vector<Type> & primal_, dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    mxArray* input[2] =
        { m_Residual.get(), primal.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 2, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::residual(primal,output)");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    primal.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::jacobian(const dotk::vector<Type> & primal_,
                                                const dotk::vector<Type> & vector_,
                                                dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    dotk::DOTk_MexArrayPtr delta_primal(mxCreateDoubleMatrix(vector_.size(), 1, mxREAL));
    vector_.gather(mxGetPr(delta_primal.get()));

    mxArray* input[3] =
        { m_Jacobian.get(), primal.get(), delta_primal.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::jacobian");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    primal.release();
    delta_primal.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::adjointJacobian(const dotk::vector<Type> & primal_,
                                                       const dotk::vector<Type> & dual_,
                                                       dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(dual_.size(), 1, mxREAL));
    dual_.gather(mxGetPr(dual.get()));

    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    mxArray* input[3] =
        { m_AdjointPartialDerivative.get(), primal.get(), dual.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::adjointJacobian");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    primal.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::hessian(const dotk::vector<Type> & primal_,
                                               const dotk::vector<Type> & dual_,
                                               const dotk::vector<Type> & vector_,
                                               dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(dual_.size(), 1, mxREAL));
    dual_.gather(mxGetPr(dual.get()));

    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    dotk::DOTk_MexArrayPtr delta_primal(mxCreateDoubleMatrix(vector_.size(), 1, mxREAL));
    vector_.gather(mxGetPr(delta_primal.get()));

    mxArray* input[4] =
        { m_Hessian.get(), primal.get(), dual.get(), delta_primal.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::hessian");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    primal.release();
    delta_primal.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::residual(const dotk::vector<Type> & state_,
                                                const dotk::vector<Type> & control_,
                                                dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[3] =
        { m_Residual.get(), state.get(), control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::residual(state,control,output)");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::partialDerivativeState(const dotk::vector<Type> & state_,
                                                              const dotk::vector<Type> & control_,
                                                              const dotk::vector<Type> & vector_,
                                                              dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    dotk::DOTk_MexArrayPtr delta_state(mxCreateDoubleMatrix(vector_.size(), 1, mxREAL));
    vector_.gather(mxGetPr(delta_state.get()));

    mxArray* input[4] =
        { m_PartialDerivativeState.get(), state.get(), control.get(), delta_state.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::partialDerivativeState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
    delta_state.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::partialDerivativeControl(const dotk::vector<Type> & state_,
                                                                const dotk::vector<Type> & control_,
                                                                const dotk::vector<Type> & vector_,
                                                                dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    dotk::DOTk_MexArrayPtr delta_control(mxCreateDoubleMatrix(vector_.size(), 1, mxREAL));
    vector_.gather(mxGetPr(delta_control.get()));

    mxArray* input[4] =
        { m_PartialDerivativeControl.get(), state.get(), control.get(), delta_control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::partialDerivativeControl");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
    delta_control.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::adjointPartialDerivativeState(const dotk::vector<Type> & state_,
                                                                     const dotk::vector<Type> & control_,
                                                                     const dotk::vector<Type> & dual_,
                                                                     dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(dual_.size(), 1, mxREAL));
    dual_.gather(mxGetPr(dual.get()));

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[4] =
        { m_AdjointPartialDerivativeState.get(), state.get(), control.get(), dual.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::adjointPartialDerivativeState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::adjointPartialDerivativeControl(const dotk::vector<Type> & state_,
                                                                       const dotk::vector<Type> & control_,
                                                                       const dotk::vector<Type> & dual_,
                                                                       dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(dual_.size(), 1, mxREAL));
    dual_.gather(mxGetPr(dual.get()));

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[4] =
        { m_AdjointPartialDerivativeControl.get(), state.get(), control.get(), dual.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::adjointPartialDerivativeControl");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::partialDerivativeStateState(const dotk::vector<Type> & state_,
                                                                   const dotk::vector<Type> & control_,
                                                                   const dotk::vector<Type> & dual_,
                                                                   const dotk::vector<Type> & vector_,
                                                                   dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(dual_.size(), 1, mxREAL));
    dual_.gather(mxGetPr(dual.get()));

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    dotk::DOTk_MexArrayPtr delta_state(mxCreateDoubleMatrix(vector_.size(), 1, mxREAL));
    vector_.gather(mxGetPr(delta_state.get()));

    mxArray* input[5] =
        { m_PartialDerivativeStateState.get(), state.get(), control.get(), dual.get(), delta_state.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 5, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::partialDerivativeStateState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
    delta_state.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::partialDerivativeStateControl(const dotk::vector<Type> & state_,
                                                                     const dotk::vector<Type> & control_,
                                                                     const dotk::vector<Type> & dual_,
                                                                     const dotk::vector<Type> & vector_,
                                                                     dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(dual_.size(), 1, mxREAL));
    dual_.gather(mxGetPr(dual.get()));

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    dotk::DOTk_MexArrayPtr delta_control(mxCreateDoubleMatrix(vector_.size(), 1, mxREAL));
    vector_.gather(mxGetPr(delta_control.get()));

    mxArray* input[5] =
        { m_PartialDerivativeStateControl.get(), state.get(), control.get(), dual.get(), delta_control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 5, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::partialDerivativeStateControl");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
    delta_control.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::partialDerivativeControlControl(const dotk::vector<Type> & state_,
                                                                       const dotk::vector<Type> & control_,
                                                                       const dotk::vector<Type> & dual_,
                                                                       const dotk::vector<Type> & vector_,
                                                                       dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(dual_.size(), 1, mxREAL));
    dual_.gather(mxGetPr(dual.get()));

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    dotk::DOTk_MexArrayPtr delta_control(mxCreateDoubleMatrix(vector_.size(), 1, mxREAL));
    vector_.gather(mxGetPr(delta_control.get()));

    mxArray* input[5] =
        { m_PartialDerivativeControlControl.get(), state.get(), control.get(), dual.get(), delta_control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 5, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::partialDerivativeControlControl");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
    delta_control.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::partialDerivativeControlState(const dotk::vector<Type> & state_,
                                                                     const dotk::vector<Type> & control_,
                                                                     const dotk::vector<Type> & dual_,
                                                                     const dotk::vector<Type> & vector_,
                                                                     dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(dual_.size(), 1, mxREAL));
    dual_.gather(mxGetPr(dual.get()));

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    dotk::DOTk_MexArrayPtr delta_state(mxCreateDoubleMatrix(vector_.size(), 1, mxREAL));
    vector_.gather(mxGetPr(delta_state.get()));

    mxArray* input[5] =
        { m_PartialDerivativeControlState.get(), state.get(), control.get(), dual.get(), delta_state.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 5, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<Type>::partialDerivativeControlState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
    delta_state.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::clear()
{
    m_Solve.release();
    m_Residual.release();
    m_Jacobian.release();
    m_Hessian.release();
    m_AdjointPartialDerivative.release();
    m_PartialDerivativeState.release();
    m_PartialDerivativeControl.release();
    m_ApplyInverseJacobianState.release();
    m_PartialDerivativeStateState.release();
    m_AdjointPartialDerivativeState.release();
    m_AdjointPartialDerivativeControl.release();
    m_PartialDerivativeStateControl.release();
    m_PartialDerivativeControlState.release();
    m_PartialDerivativeControlControl.release();
    m_ApplyAdjointInverseJacobianState.release();
}

template<class Type>
void DOTk_MexEqualityConstraint<Type>::initialize(const mxArray* operators_, const dotk::types::problem_t & type_)
{
    m_Solve.reset(mxDuplicateArray(mxGetField(operators_, 0, "solve")));
    m_Residual.reset(mxDuplicateArray(mxGetField(operators_, 0, "residual")));

    switch(type_)
    {
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_ILP:
        {
            m_Jacobian.reset(mxDuplicateArray(mxGetField(operators_, 0, "jacobian")));
            m_Hessian.reset(mxDuplicateArray(mxGetField(operators_, 0, "hessian")));
            m_AdjointPartialDerivative.reset(mxDuplicateArray(mxGetField(operators_, 0, "adjointJacobian")));
            break;
        }
        case dotk::types::TYPE_UNLP:
        case dotk::types::TYPE_NLP_BOUND:
        {
            m_PartialDerivativeState.reset(mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeState")));
            m_PartialDerivativeControl.reset(mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControl")));
            m_ApplyInverseJacobianState.reset(mxDuplicateArray(mxGetField(operators_, 0, "applyInverseJacobianState")));
            m_PartialDerivativeStateState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                           0,
                                                                           "partialDerivativeStateState")));
            m_AdjointPartialDerivativeState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                            0,
                                                                            "adjointPartialDerivativeState")));
            m_PartialDerivativeStateControl.reset(mxDuplicateArray(mxGetField(operators_,
                                                                             0,
                                                                             "partialDerivativeStateControl")));
            m_PartialDerivativeControlState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                             0,
                                                                             "partialDerivativeControlState")));
            m_AdjointPartialDerivativeControl.reset(mxDuplicateArray(mxGetField(operators_,
                                                                              0,
                                                                              "adjointPartialDerivativeControl")));
            m_PartialDerivativeControlControl.reset(mxDuplicateArray(mxGetField(operators_,
                                                                               0,
                                                                               "partialDerivativeControlControl")));
            m_ApplyAdjointInverseJacobianState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                                 0,
                                                                                 "applyAdjointInverseJacobianState")));
            break;
        }
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::TYPE_CNLP:
        {
            m_PartialDerivativeState.reset(mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeState")));
            m_PartialDerivativeControl.reset(mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControl")));
            m_ApplyInverseJacobianState.reset(mxDuplicateArray(mxGetField(operators_, 0, "applyInverseJacobianState")));
            m_PartialDerivativeStateState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                           0,
                                                                           "partialDerivativeStateState")));
            m_AdjointPartialDerivativeState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                            0,
                                                                            "adjointPartialDerivativeState")));
            m_PartialDerivativeStateControl.reset(mxDuplicateArray(mxGetField(operators_,
                                                                             0,
                                                                             "partialDerivativeStateControl")));
            m_PartialDerivativeControlState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                             0,
                                                                             "partialDerivativeControlState")));
            m_AdjointPartialDerivativeControl.reset(mxDuplicateArray(mxGetField(operators_,
                                                                              0,
                                                                              "adjointPartialDerivativeControl")));
            m_PartialDerivativeControlControl.reset(mxDuplicateArray(mxGetField(operators_,
                                                                               0,
                                                                               "partialDerivativeControlControl")));
            m_ApplyAdjointInverseJacobianState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                                 0,
                                                                                 "applyAdjointInverseJacobianState")));
            break;
        }
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::string err("\nERROR: Invalid Problem Type in Call to DOTk_MexEqualityConstraint<Type>::initialize\n");
            mexErrMsgTxt(err.c_str());
            break;
        }
    }
}

}
