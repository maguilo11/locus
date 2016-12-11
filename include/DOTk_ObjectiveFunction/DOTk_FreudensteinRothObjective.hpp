/*
 * DOTk_FreudensteinRothObjective.hpp
 *
 *  Created on: May 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_FREUDENSTEINROTHOBJECTIVE_HPP_
#define DOTK_FREUDENSTEINROTHOBJECTIVE_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_FreudensteinRothObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_FreudensteinRothObjective();
    virtual ~DOTk_FreudensteinRothObjective();

    virtual Real value(const dotk::Vector<Real> & primal_);
    virtual void gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & gradient_);
    virtual void hessian(const dotk::Vector<Real> & primal_,
                         const dotk::Vector<Real> & vector_,
                         dotk::Vector<Real> & output_);

private:
    DOTk_FreudensteinRothObjective(const dotk::DOTk_FreudensteinRothObjective&);
    dotk::DOTk_FreudensteinRothObjective operator=(const dotk::DOTk_FreudensteinRothObjective&);
};

}

#endif /* DOTK_FREUDENSTEINROTHOBJECTIVE_HPP_ */
