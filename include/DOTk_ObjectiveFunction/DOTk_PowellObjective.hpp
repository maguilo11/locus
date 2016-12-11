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

template<typename ScalarType>
class Vector;

class DOTk_PowellObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_PowellObjective();
    virtual ~DOTk_PowellObjective();

    virtual Real value(const dotk::Vector<Real> & primal_);
    virtual void gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & output_);
    virtual void hessian(const dotk::Vector<Real> & primal_,
                         const dotk::Vector<Real> & vector_,
                         dotk::Vector<Real> & output_);

private:
    DOTk_PowellObjective(const dotk::DOTk_PowellObjective&);
    dotk::DOTk_PowellObjective operator=(const dotk::DOTk_PowellObjective&);
};

}

#endif /* DOTK_POWELLOBJECTIVE_HPP_ */
