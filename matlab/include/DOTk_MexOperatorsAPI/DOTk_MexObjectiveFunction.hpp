/*
 * DOTk_MexObjectiveFunction.hpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXOBJECTIVEFUNCTION_HPP_
#define DOTK_MEXOBJECTIVEFUNCTION_HPP_

#include <mex.h>

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_MexArrayPtr;

template<typename ScalarType>
class DOTk_MexObjectiveFunction : public DOTk_ObjectiveFunction<ScalarType>
{
public:
    DOTk_MexObjectiveFunction(const mxArray* operators_, const dotk::types::problem_t & type_);
    virtual ~DOTk_MexObjectiveFunction();

    ScalarType value(const dotk::Vector<ScalarType> & primal_);
    void gradient(const dotk::Vector<ScalarType> & primal_, dotk::Vector<ScalarType> & output_);
    void hessian(const dotk::Vector<ScalarType> & primal_,
                 const dotk::Vector<ScalarType> & delta_primal_,
                 dotk::Vector<ScalarType> & output_);

    ScalarType value(const dotk::Vector<ScalarType> & state_, const dotk::Vector<ScalarType> & control_);
    void partialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                const dotk::Vector<ScalarType> & control_,
                                dotk::Vector<ScalarType> & output_);
    void partialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                  const dotk::Vector<ScalarType> & control_,
                                  dotk::Vector<ScalarType> & output_);
    void partialDerivativeStateState(const dotk::Vector<ScalarType> & state_,
                                     const dotk::Vector<ScalarType> & control_,
                                     const dotk::Vector<ScalarType> & vector_,
                                     dotk::Vector<ScalarType> & output_);
    void partialDerivativeStateControl(const dotk::Vector<ScalarType> & state_,
                                       const dotk::Vector<ScalarType> & control_,
                                       const dotk::Vector<ScalarType> & vector_,
                                       dotk::Vector<ScalarType> & output_);
    void partialDerivativeControlControl(const dotk::Vector<ScalarType> & state_,
                                         const dotk::Vector<ScalarType> & control_,
                                         const dotk::Vector<ScalarType> & vector_,
                                         dotk::Vector<ScalarType> & output_);
    void partialDerivativeControlState(const dotk::Vector<ScalarType> & state_,
                                       const dotk::Vector<ScalarType> & control_,
                                       const dotk::Vector<ScalarType> & vector_,
                                       dotk::Vector<ScalarType> & output_);

private:
    void clear();
    void initialize(const mxArray* operators_, const dotk::types::problem_t & type_);

private:
    dotk::DOTk_MexArrayPtr m_Value;
    dotk::DOTk_MexArrayPtr m_FirstDerivative;
    dotk::DOTk_MexArrayPtr m_SecondDerivative;
    dotk::DOTk_MexArrayPtr m_FirstDerivativeState;
    dotk::DOTk_MexArrayPtr m_FirstDerivativeControl;
    dotk::DOTk_MexArrayPtr m_SecondDerivativeStateState;
    dotk::DOTk_MexArrayPtr m_SecondDerivativeStateControl;
    dotk::DOTk_MexArrayPtr m_SecondDerivativeControlState;
    dotk::DOTk_MexArrayPtr m_SecondDerivativeControlControl;

private:
    DOTk_MexObjectiveFunction(const dotk::DOTk_MexObjectiveFunction<ScalarType> &);
    dotk::DOTk_MexObjectiveFunction<ScalarType> & operator=(const dotk::DOTk_MexObjectiveFunction<ScalarType> &);
};

}

#endif /* DOTK_MEXOBJECTIVEFUNCTION_HPP_ */
