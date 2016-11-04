/*
 * DOTk_NocedalAndWrightObjective.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NOCEDALANDWRIGHTOBJECTIVE_HPP_
#define DOTK_NOCEDALANDWRIGHTOBJECTIVE_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<class Type>
class vector;

class DOTk_NocedalAndWrightObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_NocedalAndWrightObjective();
    virtual ~DOTk_NocedalAndWrightObjective();

    virtual Real value(const dotk::vector<Real> & primal_);
    virtual void gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & output_);
    virtual void hessian(const dotk::vector<Real> & primal_,
                         const dotk::vector<Real> & vector_,
                         dotk::vector<Real> & output_);

private:
    // unimplemented
    DOTk_NocedalAndWrightObjective(const dotk::DOTk_NocedalAndWrightObjective&);
    dotk::DOTk_NocedalAndWrightObjective operator=(const dotk::DOTk_NocedalAndWrightObjective&);

};

}

#endif /* DOTK_NOCEDALANDWRIGHTOBJECTIVE_HPP_ */
