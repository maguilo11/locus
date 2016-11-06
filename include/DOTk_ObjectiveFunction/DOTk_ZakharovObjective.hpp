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

template<typename Type>
class vector;

class DOTk_ZakharovObjective : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    explicit DOTk_ZakharovObjective(const dotk::vector<Real> & input_);
    virtual ~DOTk_ZakharovObjective();

    Real value(const dotk::vector<Real> & primal_);
    void gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & output_);
    void hessian(const dotk::vector<Real> & primal_, const dotk::vector<Real> & vector_, dotk::vector<Real> & output_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_Data;

private:
    DOTk_ZakharovObjective(const dotk::DOTk_ZakharovObjective&);
    dotk::DOTk_ZakharovObjective operator=(const dotk::DOTk_ZakharovObjective&);
};

}

#endif /* DOTK_ZAKHAROVOBJECTIVE_HPP_ */
