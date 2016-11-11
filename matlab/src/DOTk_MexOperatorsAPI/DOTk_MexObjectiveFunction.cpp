/*
 * DOTk_MexObjectiveFunction.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexObjectiveFunction.hpp"

namespace dotk
{

template<class Type>
DOTk_MexObjectiveFunction<Type>::DOTk_MexObjectiveFunction(const mxArray* operators_,
                                                           const dotk::types::problem_t & type_) :
        m_Value(NULL),
        m_FirstDerivative(NULL),
        m_SecondDerivative(NULL),
        m_FirstDerivativeState(NULL),
        m_FirstDerivativeControl(NULL),
        m_SecondDerivativeStateState(NULL),
        m_SecondDerivativeStateControl(NULL),
        m_SecondDerivativeControlState(NULL),
        m_SecondDerivativeControlControl(NULL)
{
    this->initialize(operators_, type_);
}

template<class Type>
DOTk_MexObjectiveFunction<Type>::~DOTk_MexObjectiveFunction()
{
    this->clear();
}

template<class Type>
Type DOTk_MexObjectiveFunction<Type>::value(const dotk::vector<Type> & primal_)
{
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    mxArray* input[2] =
        { m_Value.get(), primal.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 2, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::value(control)");
    double alpha = mxGetScalar(output[0]);

    primal.release();

    return (alpha);
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::gradient(const dotk::vector<Type> & primal_, dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    mxArray* input[2] =
        { m_FirstDerivative.get(), primal.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 2, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::gradient");
    dotk::mex::copyData(primal_.size(), mxGetPr(output[0]), output_);

    primal.release();
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::hessian(const dotk::vector<Type> & primal_,
                                              const dotk::vector<Type> & delta_primal_,
                                              dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    dotk::DOTk_MexArrayPtr delta_primal(mxCreateDoubleMatrix(delta_primal_.size(), 1, mxREAL));
    delta_primal_.gather(mxGetPr(delta_primal.get()));

    mxArray* input[3] =
        { m_SecondDerivative.get(), primal.get(), delta_primal.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::hessian");
    dotk::mex::copyData(primal_.size(), mxGetPr(output[0]), output_);

    primal.release();
    delta_primal.release();
}

template<class Type>
Type DOTk_MexObjectiveFunction<Type>::value(const dotk::vector<Type> & state_, const dotk::vector<Type> & control_)
{
    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[3] =
        { m_Value.get(), state.get(), control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::value(state,control)");
    double value = mxGetScalar(output[0]);

    state.release();
    control.release();

    return (value);
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::partialDerivativeState(const dotk::vector<Type> & state_,
                                                             const dotk::vector<Type> & control_,
                                                             dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[3] =
        { m_FirstDerivativeState.get(), state.get(), control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::partialDerivativeState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::partialDerivativeControl(const dotk::vector<Type> & state_,
                                                               const dotk::vector<Type> & control_,
                                                               dotk::vector<Type> & output_)
{
    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[3] =
        { m_FirstDerivativeControl.get(), state.get(), control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::partialDerivativeControl");
    dotk::mex::copyData(control_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::partialDerivativeStateState(const dotk::vector<Type> & state_,
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
        { m_SecondDerivativeStateState.get(), state.get(), control.get(), delta_state.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::partialDerivativeStateState");
    dotk::mex::copyData(output_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
    delta_state.release();
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::partialDerivativeStateControl(const dotk::vector<Type> & state_,
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
        { m_SecondDerivativeStateControl.get(), state.get(), control.get(), delta_control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::partialDerivativeStateControl");
    dotk::mex::copyData(state_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
    delta_control.release();
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::partialDerivativeControlState(const dotk::vector<Type> & state_,
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
        { m_SecondDerivativeControlState.get(), state.get(), control.get(), delta_state.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::partialDerivativeControlState");
    dotk::mex::copyData(control_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
    delta_state.release();
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::partialDerivativeControlControl(const dotk::vector<Type> & state_,
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
        { m_SecondDerivativeControlControl.get(), state.get(), control.get(), delta_control.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err,
                               "ERROR: Invalid Call to DOTk_MexObjectiveFunction<T>::partialDerivativeControlControl");
    dotk::mex::copyData(control_.size(), mxGetPr(output[0]), output_);

    state.release();
    control.release();
    delta_control.release();
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::clear()
{
    m_Value.release();
    m_FirstDerivative.release();
    m_SecondDerivative.release();
    m_FirstDerivativeState.release();
    m_FirstDerivativeControl.release();
    m_SecondDerivativeStateState.release();
    m_SecondDerivativeStateControl.release();
    m_SecondDerivativeControlState.release();
    m_SecondDerivativeControlControl.release();
}

template<class Type>
void DOTk_MexObjectiveFunction<Type>::initialize(const mxArray* operators_, const dotk::types::problem_t & type_)
{
    m_Value.reset(mxDuplicateArray(mxGetField(operators_, 0, "value")));
    switch(type_)
    {
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_ILP:
        {
            m_FirstDerivative.reset(mxDuplicateArray(mxGetField(operators_, 0, "gradient")));
            m_SecondDerivative.reset(mxDuplicateArray(mxGetField(operators_, 0, "hessian")));
            break;
        }
        case dotk::types::TYPE_UNLP:
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_NLP_BOUND:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::TYPE_CNLP:
        {
            m_FirstDerivativeState.reset(mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeState")));
            m_FirstDerivativeControl.reset(mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControl")));
            m_SecondDerivativeStateState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                           0,
                                                                           "partialDerivativeStateState")));
            m_SecondDerivativeStateControl.reset(mxDuplicateArray(mxGetField(operators_,
                                                                             0,
                                                                             "partialDerivativeStateControl")));
            m_SecondDerivativeControlState.reset(mxDuplicateArray(mxGetField(operators_,
                                                                             0,
                                                                             "partialDerivativeControlState")));
            m_SecondDerivativeControlControl.reset(mxDuplicateArray(mxGetField(operators_,
                                                                               0,
                                                                               "partialDerivativeControlControl")));
            break;
        }
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::string err("\nERROR: Invalid Problem Type in Call to DOTk_MexObjectiveFunction<T>::initialize\n");
            mexErrMsgTxt(err.c_str());
            break;
        }
    }
}

}