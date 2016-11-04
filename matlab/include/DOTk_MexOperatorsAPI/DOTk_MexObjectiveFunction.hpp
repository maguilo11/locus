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

template<class Type>
class vector;

class DOTk_MexArrayPtr;

template<class Type>
class DOTk_MexObjectiveFunction : public DOTk_ObjectiveFunction<Type>
{
public:
    DOTk_MexObjectiveFunction(const mxArray* operators_, const dotk::types::problem_t & type_);
    virtual ~DOTk_MexObjectiveFunction();

    Type value(const dotk::vector<Type> & primal_);
    void gradient(const dotk::vector<Type> & primal_, dotk::vector<Type> & output_);
    void hessian(const dotk::vector<Type> & primal_,
                 const dotk::vector<Type> & delta_primal_,
                 dotk::vector<Type> & output_);

    Type value(const dotk::vector<Type> & state_, const dotk::vector<Type> & control_);
    void partialDerivativeState(const dotk::vector<Type> & state_,
                                const dotk::vector<Type> & control_,
                                dotk::vector<Type> & output_);
    void partialDerivativeControl(const dotk::vector<Type> & state_,
                                  const dotk::vector<Type> & control_,
                                  dotk::vector<Type> & output_);
    void partialDerivativeStateState(const dotk::vector<Type> & state_,
                                     const dotk::vector<Type> & control_,
                                     const dotk::vector<Type> & vector_,
                                     dotk::vector<Type> & output_);
    void partialDerivativeStateControl(const dotk::vector<Type> & state_,
                                       const dotk::vector<Type> & control_,
                                       const dotk::vector<Type> & vector_,
                                       dotk::vector<Type> & output_);
    void partialDerivativeControlControl(const dotk::vector<Type> & state_,
                                         const dotk::vector<Type> & control_,
                                         const dotk::vector<Type> & vector_,
                                         dotk::vector<Type> & output_);
    void partialDerivativeControlState(const dotk::vector<Type> & state_,
                                       const dotk::vector<Type> & control_,
                                       const dotk::vector<Type> & vector_,
                                       dotk::vector<Type> & output_);

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
    DOTk_MexObjectiveFunction(const dotk::DOTk_MexObjectiveFunction<Type> &);
    dotk::DOTk_MexObjectiveFunction<Type> & operator=(const dotk::DOTk_MexObjectiveFunction<Type> &);
};

}

#endif /* DOTK_MEXOBJECTIVEFUNCTION_HPP_ */
