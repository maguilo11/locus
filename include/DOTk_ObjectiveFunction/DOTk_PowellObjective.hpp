/*
 * DOTk_PowellObjective.hpp
 *
 *  Created on: May 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_POWELLOBJECTIVE_HPP_
#define DOTK_POWELLOBJECTIVE_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<class Type>
class vector;

class DOTk_PowellObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_PowellObjective();
    virtual ~DOTk_PowellObjective();

    virtual Real value(const dotk::vector<Real> & primal_);
    virtual void gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & output_);
    virtual void hessian(const dotk::vector<Real> & primal_,
                         const dotk::vector<Real> & vector_,
                         dotk::vector<Real> & output_);

private:
    DOTk_PowellObjective(const dotk::DOTk_PowellObjective&);
    dotk::DOTk_PowellObjective operator=(const dotk::DOTk_PowellObjective&);
};

}

#endif /* DOTK_POWELLOBJECTIVE_HPP_ */
