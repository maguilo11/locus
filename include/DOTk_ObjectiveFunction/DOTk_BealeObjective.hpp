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

template<typename Type>
class vector;

class DOTk_BealeObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_BealeObjective();
    virtual ~DOTk_BealeObjective();

    virtual Real value(const dotk::vector<Real> & primal_);
    virtual void gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & derivative_);
    virtual void hessian(const dotk::vector<Real> & primal_,
                         const dotk::vector<Real> & delta_primal_,
                         dotk::vector<Real> & hessian_times_delta_primal_);

private:
    DOTk_BealeObjective(const dotk::DOTk_BealeObjective&);
    dotk::DOTk_BealeObjective operator=(const dotk::DOTk_BealeObjective&);
};

}

#endif /* DOTK_BEALEOBJECTIVE_HPP_ */
