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

class DOTk_MexObjectiveFunction : public DOTk_ObjectiveFunction<double>
{
public:
    DOTk_MexObjectiveFunction(const mxArray* operators_, const dotk::types::problem_t & type_);
    virtual ~DOTk_MexObjectiveFunction();

    double value(const dotk::Vector<double> & control_);
    void gradient(const dotk::Vector<double> & control_, dotk::Vector<double> & output_);
    void hessian(const dotk::Vector<double> & control_,
                 const dotk::Vector<double> & vector_,
                 dotk::Vector<double> & output_);

    double value(const dotk::Vector<double> & state_, const dotk::Vector<double> & control_);
    void partialDerivativeState(const dotk::Vector<double> & state_,
                                const dotk::Vector<double> & control_,
                                dotk::Vector<double> & output_);
    void partialDerivativeControl(const dotk::Vector<double> & state_,
                                  const dotk::Vector<double> & control_,
                                  dotk::Vector<double> & output_);
    void partialDerivativeStateState(const dotk::Vector<double> & state_,
                                     const dotk::Vector<double> & control_,
                                     const dotk::Vector<double> & vector_,
                                     dotk::Vector<double> & output_);
    void partialDerivativeStateControl(const dotk::Vector<double> & state_,
                                       const dotk::Vector<double> & control_,
                                       const dotk::Vector<double> & vector_,
                                       dotk::Vector<double> & output_);
    void partialDerivativeControlControl(const dotk::Vector<double> & state_,
                                         const dotk::Vector<double> & control_,
                                         const dotk::Vector<double> & vector_,
                                         dotk::Vector<double> & output_);
    void partialDerivativeControlState(const dotk::Vector<double> & state_,
                                       const dotk::Vector<double> & control_,
                                       const dotk::Vector<double> & vector_,
                                       dotk::Vector<double> & output_);

private:
    void clear();
    void initialize(const mxArray* operators_, const dotk::types::problem_t & type_);

private:
    mxArray* m_Value;
    mxArray* m_Gradient;
    mxArray* m_Hessian;
    mxArray* m_PartialDerivativeState;
    mxArray* m_PartialDerivativeControl;
    mxArray* m_PartialDerivativeStateState;
    mxArray* m_PartialDerivativeStateControl;
    mxArray* m_PartialDerivativeControlState;
    mxArray* m_PartialDerivativeControlControl;

private:
    DOTk_MexObjectiveFunction(const dotk::DOTk_MexObjectiveFunction &);
    dotk::DOTk_MexObjectiveFunction & operator=(const dotk::DOTk_MexObjectiveFunction &);
};

}

#endif /* DOTK_MEXOBJECTIVEFUNCTION_HPP_ */
