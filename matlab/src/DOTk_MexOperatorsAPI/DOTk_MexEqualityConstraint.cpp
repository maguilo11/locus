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

template<typename ScalarType>
DOTk_MexEqualityConstraint<ScalarType>::DOTk_MexEqualityConstraint(const mxArray* operators_,
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

template<typename ScalarType>
DOTk_MexEqualityConstraint<ScalarType>::~DOTk_MexEqualityConstraint()
{
    this->clear();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::solve(const dotk::Vector<ScalarType> & control_, dotk::Vector<ScalarType> & output_)
{
    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[2] =
        { m_Solve.get(), control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 2, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::solve");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    control.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::applyInverseJacobianState(const dotk::Vector<ScalarType> & state_,
                                                                 const dotk::Vector<ScalarType> & control_,
                                                                 const dotk::Vector<ScalarType> & rhs_,
                                                                 dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::applyInverseJacobianState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    rhs.release();
    state.release();
    control.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::applyAdjointInverseJacobianState(const dotk::Vector<ScalarType> & state_,
                                                                        const dotk::Vector<ScalarType> & control_,
                                                                        const dotk::Vector<ScalarType> & rhs_,
                                                                        dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::applyAdjointInverseJacobianState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    rhs.release();
    state.release();
    control.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::residual(const dotk::Vector<ScalarType> & primal_, dotk::Vector<ScalarType> & output_)
{
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    mxArray* input[2] =
        { m_Residual.get(), primal.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 2, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::residual(primal,output)");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    primal.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::jacobian(const dotk::Vector<ScalarType> & primal_,
                                                const dotk::Vector<ScalarType> & vector_,
                                                dotk::Vector<ScalarType> & output_)
{
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    dotk::DOTk_MexArrayPtr delta_primal(mxCreateDoubleMatrix(vector_.size(), 1, mxREAL));
    vector_.gather(mxGetPr(delta_primal.get()));

    mxArray* input[3] =
        { m_Jacobian.get(), primal.get(), delta_primal.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::jacobian");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    primal.release();
    delta_primal.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::adjointJacobian(const dotk::Vector<ScalarType> & primal_,
                                                       const dotk::Vector<ScalarType> & dual_,
                                                       dotk::Vector<ScalarType> & output_)
{
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(dual_.size(), 1, mxREAL));
    dual_.gather(mxGetPr(dual.get()));

    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    mxArray* input[3] =
        { m_AdjointPartialDerivative.get(), primal.get(), dual.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::adjointJacobian");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    primal.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::hessian(const dotk::Vector<ScalarType> & primal_,
                                               const dotk::Vector<ScalarType> & dual_,
                                               const dotk::Vector<ScalarType> & vector_,
                                               dotk::Vector<ScalarType> & output_)
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
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::hessian");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    primal.release();
    delta_primal.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::residual(const dotk::Vector<ScalarType> & state_,
                                                const dotk::Vector<ScalarType> & control_,
                                                dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::residual(state,control,output)");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                                              const dotk::Vector<ScalarType> & control_,
                                                              const dotk::Vector<ScalarType> & vector_,
                                                              dotk::Vector<ScalarType> & output_)
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
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
    delta_state.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                                                const dotk::Vector<ScalarType> & control_,
                                                                const dotk::Vector<ScalarType> & vector_,
                                                                dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeControl");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
    delta_control.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::adjointPartialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                                                     const dotk::Vector<ScalarType> & control_,
                                                                     const dotk::Vector<ScalarType> & dual_,
                                                                     dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::adjointPartialDerivativeState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::adjointPartialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                                                       const dotk::Vector<ScalarType> & control_,
                                                                       const dotk::Vector<ScalarType> & dual_,
                                                                       dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::adjointPartialDerivativeControl");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeStateState(const dotk::Vector<ScalarType> & state_,
                                                                   const dotk::Vector<ScalarType> & control_,
                                                                   const dotk::Vector<ScalarType> & dual_,
                                                                   const dotk::Vector<ScalarType> & vector_,
                                                                   dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeStateState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
    delta_state.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeStateControl(const dotk::Vector<ScalarType> & state_,
                                                                     const dotk::Vector<ScalarType> & control_,
                                                                     const dotk::Vector<ScalarType> & dual_,
                                                                     const dotk::Vector<ScalarType> & vector_,
                                                                     dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeStateControl");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
    delta_control.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeControlControl(const dotk::Vector<ScalarType> & state_,
                                                                       const dotk::Vector<ScalarType> & control_,
                                                                       const dotk::Vector<ScalarType> & dual_,
                                                                       const dotk::Vector<ScalarType> & vector_,
                                                                       dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeControlControl");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
    delta_control.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeControlState(const dotk::Vector<ScalarType> & state_,
                                                                     const dotk::Vector<ScalarType> & control_,
                                                                     const dotk::Vector<ScalarType> & dual_,
                                                                     const dotk::Vector<ScalarType> & vector_,
                                                                     dotk::Vector<ScalarType> & output_)
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
                               "ERROR: Invalid Call to DOTk_MexEqualityConstraint<ScalarType>::partialDerivativeControlState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    dual.release();
    state.release();
    control.release();
    delta_state.release();
}

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::clear()
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

template<typename ScalarType>
void DOTk_MexEqualityConstraint<ScalarType>::initialize(const mxArray* operators_, const dotk::types::problem_t & type_)
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
            std::string err("\nERROR: Invalid Problem ScalarType in Call to DOTk_MexEqualityConstraint<ScalarType>::initialize\n");
            mexErrMsgTxt(err.c_str());
            break;
        }
    }
}

}
