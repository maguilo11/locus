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

template<typename ScalarType>
class Vector;

class DOTk_NocedalAndWrightObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_NocedalAndWrightObjective();
    virtual ~DOTk_NocedalAndWrightObjective();

    virtual Real value(const dotk::Vector<Real> & primal_);
    virtual void gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & output_);
    virtual void hessian(const dotk::Vector<Real> & primal_,
                         const dotk::Vector<Real> & vector_,
                         dotk::Vector<Real> & output_);

private:
    // unimplemented
    DOTk_NocedalAndWrightObjective(const dotk::DOTk_NocedalAndWrightObjective&);
    dotk::DOTk_NocedalAndWrightObjective operator=(const dotk::DOTk_NocedalAndWrightObjective&);

};

}

#endif /* DOTK_NOCEDALANDWRIGHTOBJECTIVE_HPP_ */
