/*
 * DOTk_ZakharovObjective.hpp
 *
 *  Created on: May 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ZAKHAROVOBJECTIVE_HPP_
#define DOTK_ZAKHAROVOBJECTIVE_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_ZakharovObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    explicit DOTk_ZakharovObjective(const dotk::Vector<Real> & input_);
    virtual ~DOTk_ZakharovObjective();

    Real value(const dotk::Vector<Real> & primal_);
    void gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & output_);
    void hessian(const dotk::Vector<Real> & primal_, const dotk::Vector<Real> & vector_, dotk::Vector<Real> & output_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_Data;

private:
    DOTk_ZakharovObjective(const dotk::DOTk_ZakharovObjective&);
    dotk::DOTk_ZakharovObjective operator=(const dotk::DOTk_ZakharovObjective&);
};

}

#endif /* DOTK_ZAKHAROVOBJECTIVE_HPP_ */
