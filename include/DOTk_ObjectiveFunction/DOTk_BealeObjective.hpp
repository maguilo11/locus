/*
 * DOTk_BealeObjective.hpp
 *
 *  Created on: May 30, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_BEALEOBJECTIVE_HPP_
#define DOTK_BEALEOBJECTIVE_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_BealeObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_BealeObjective();
    virtual ~DOTk_BealeObjective();

    virtual Real value(const dotk::Vector<Real> & primal_);
    virtual void gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & derivative_);
    virtual void hessian(const dotk::Vector<Real> & primal_,
                         const dotk::Vector<Real> & delta_primal_,
                         dotk::Vector<Real> & hessian_times_delta_primal_);

private:
    DOTk_BealeObjective(const dotk::DOTk_BealeObjective&);
    dotk::DOTk_BealeObjective operator=(const dotk::DOTk_BealeObjective&);
};

}

#endif /* DOTK_BEALEOBJECTIVE_HPP_ */
