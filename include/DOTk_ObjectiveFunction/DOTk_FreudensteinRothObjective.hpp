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

template<class Type>
class vector;

class DOTk_FreudensteinRothObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_FreudensteinRothObjective();
    virtual ~DOTk_FreudensteinRothObjective();

    virtual Real value(const dotk::vector<Real> & primal_);
    virtual void gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & gradient_);
    virtual void hessian(const dotk::vector<Real> & primal_,
                         const dotk::vector<Real> & vector_,
                         dotk::vector<Real> & output_);

private:
    DOTk_FreudensteinRothObjective(const dotk::DOTk_FreudensteinRothObjective&);
    dotk::DOTk_FreudensteinRothObjective operator=(const dotk::DOTk_FreudensteinRothObjective&);
};

}

#endif /* DOTK_FREUDENSTEINROTHOBJECTIVE_HPP_ */
