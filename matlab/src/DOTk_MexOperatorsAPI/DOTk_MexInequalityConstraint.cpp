/*
 * DOTk_MexInequalityConstraint.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexInequalityConstraint.hpp"

namespace dotk
{

template<typename ScalarType>
DOTk_MexInequalityConstraint<ScalarType>::DOTk_MexInequalityConstraint(const mxArray* operators_,
                                                                 const dotk::types::problem_t & type_) :
        dotk::DOTk_InequalityConstraint<double>(),
        m_Value(NULL),
        m_Evaluate(NULL),
        m_FirstDerivative(NULL),
        m_SecondDerivative(NULL),
        m_FirstDerivativeWrtState(NULL),
        m_FirstDerivativeWrtControl(NULL)
{
    this->initialize(operators_, type_);
}

template<typename ScalarType>
DOTk_MexInequalityConstraint<ScalarType>::~DOTk_MexInequalityConstraint()
{
    this->clear();
}

template<typename ScalarType>
ScalarType DOTk_MexInequalityConstraint<ScalarType>::bound(const size_t & index_)
{
    dotk::DOTk_MexArrayPtr index(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(index.get()))[0] = index_;

    mxArray* input[2] =
        { m_Value.get(), index.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 2, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexInequalityConstraint::bound");
    double bound = mxGetScalar(output[0]);

    index.release();

    return (bound);
}

template<typename ScalarType>
ScalarType DOTk_MexInequalityConstraint<ScalarType>::value(const dotk::Vector<ScalarType> & primal_, const size_t & index_)
{
    dotk::DOTk_MexArrayPtr index(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(index.get()))[0] = index_;

    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    mxArray* input[3] =
        { m_Evaluate.get(), primal.get(), index.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexInequalityConstraint::value");
    double bound = mxGetScalar(output[0]);

    index.release();
    primal.release();

    return (bound);
}

template<typename ScalarType>
void DOTk_MexInequalityConstraint<ScalarType>::gradient(const dotk::Vector<ScalarType> & primal_,
                                                  const size_t & index_,
                                                  dotk::Vector<ScalarType> & derivative_)
{
    dotk::DOTk_MexArrayPtr index(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(index.get()))[0] = index_;

    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    mxArray* input[3] =
        { m_FirstDerivative.get(), primal.get(), index.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 3, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexInequalityConstraint::gradient");
    dotk::mex::copyData(primal_.size(), mxGetPr(output[0]), derivative_);

    index.release();
    primal.release();
}

template<typename ScalarType>
void DOTk_MexInequalityConstraint<ScalarType>::hessian(const dotk::Vector<ScalarType> & primal_,
                                                 const dotk::Vector<ScalarType> & delta_primal_,
                                                 const size_t & index_,
                                                 dotk::Vector<ScalarType> & derivative_)
{
    dotk::DOTk_MexArrayPtr index(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(index.get()))[0] = index_;

    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(primal_.size(), 1, mxREAL));
    primal_.gather(mxGetPr(primal.get()));

    dotk::DOTk_MexArrayPtr delta_primal(mxCreateDoubleMatrix(delta_primal_.size(), 1, mxREAL));
    delta_primal_.gather(mxGetPr(delta_primal.get()));

    mxArray* input[4] =
        { m_SecondDerivative.get(), primal.get(), delta_primal.get(), index.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexInequalityConstraint::hessian");
    dotk::mex::copyData(primal_.size(), mxGetPr(output[0]), derivative_);

    index.release();
    primal.release();
    delta_primal.release();
}

template<typename ScalarType>
ScalarType DOTk_MexInequalityConstraint<ScalarType>::value(const dotk::Vector<ScalarType> & state_,
                                               const dotk::Vector<ScalarType> & control_,
                                               const size_t & index_)
{
    dotk::DOTk_MexArrayPtr index(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(index.get()))[0] = index_;

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[4] =
        { m_Evaluate.get(), state.get(), control.get(), index.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexInequalityConstraint::value ScalarType NLP");
    double bound = mxGetScalar(output[0]);

    index.release();
    state.release();
    control.release();

    return (bound);
}

template<typename ScalarType>
void DOTk_MexInequalityConstraint<ScalarType>::partialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                                                const dotk::Vector<ScalarType> & control_,
                                                                const size_t & index_,
                                                                dotk::Vector<ScalarType> & derivative_)
{
    dotk::DOTk_MexArrayPtr index(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(index.get()))[0] = index_;

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[4] =
        { m_FirstDerivativeWrtState.get(), state.get(), control.get(), index.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexInequalityConstraint::partialDerivativeState");
    dotk::mex::copyData(state_.size(), mxGetPr(output[0]), derivative_);

    index.release();
    state.release();
    control.release();
}

template<typename ScalarType>
void DOTk_MexInequalityConstraint<ScalarType>::partialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                                                  const dotk::Vector<ScalarType> & control_,
                                                                  const size_t & index_,
                                                                  dotk::Vector<ScalarType> & derivative_)
{
    dotk::DOTk_MexArrayPtr index(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(index.get()))[0] = index_;

    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(state_.size(), 1, mxREAL));
    state_.gather(mxGetPr(state.get()));

    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(control_.size(), 1, mxREAL));
    control_.gather(mxGetPr(control.get()));

    mxArray* input[4] =
        { m_FirstDerivativeWrtControl.get(), state.get(), control.get(), index.get() };
    mxArray* output[1];

    mxArray* err = mexCallMATLABWithTrap(1, output, 4, input, "feval");
    dotk::mex::handleException(err, "ERROR: Invalid Call to DOTk_MexInequalityConstraint::partialDerivativeControl");
    dotk::mex::copyData(control_.size(), mxGetPr(output[0]), derivative_);

    index.release();
    state.release();
    control.release();
}

template<typename ScalarType>
void DOTk_MexInequalityConstraint<ScalarType>::clear()
{
    m_Value.release();
    m_Evaluate.release();
    m_FirstDerivative.release();
    m_SecondDerivative.release();
    m_FirstDerivativeWrtState.release();
    m_FirstDerivativeWrtControl.release();
}

template<typename ScalarType>
void DOTk_MexInequalityConstraint<ScalarType>::initialize(const mxArray* operators_, const dotk::types::problem_t & type_)
{
    m_Value.reset(mxDuplicateArray(mxGetField(operators_, 0, "bound")));
    m_Evaluate.reset(mxDuplicateArray(mxGetField(operators_, 0, "value")));

    switch(type_)
    {
        case dotk::types::TYPE_ILP:
        case dotk::types::TYPE_CLP:
        {
            m_FirstDerivative.reset(mxDuplicateArray(mxGetField(operators_, 0, "gradient")));
            break;
        }
        case dotk::types::TYPE_CNLP:
        {
            m_FirstDerivativeWrtState.reset(mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeState")));
            m_FirstDerivativeWrtControl.reset(mxDuplicateArray(mxGetField(operators_, 0, "partialDerivativeControl")));
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
            std::string err("\nERROR: Invalid Problem ScalarType in Call to DOTk_MexInequalityConstraint::initialize\n");
            mexErrMsgTxt(err.c_str());
            break;
        }
    }
}

}
