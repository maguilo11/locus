/*
 * DOTk_GradientTypeUNP.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_GRADIENTTYPEUNP_HPP_
#define DOTK_GRADIENTTYPEUNP_HPP_

#include "DOTk_Functor.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"

namespace dotk
{

class DOTk_Primal;

template<class Type>
class vector;

class DOTk_GradientTypeUNP : public DOTk_Functor
{
public:
    DOTk_GradientTypeUNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                         const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                         const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~DOTk_GradientTypeUNP();

    virtual void operator()(const dotk::vector<Real> & control_, dotk::vector<Real> & gradient_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    void allocate(dotk::types::variable_t type_, const std::tr1::shared_ptr<dotk::vector<Real> > & data_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_Dual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_State;
    std::tr1::shared_ptr<dotk::vector<Real> > m_StateWorkVec;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ControlWorkVec;

    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;
    std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > m_EqualityContraint;

private:
    DOTk_GradientTypeUNP(const dotk::DOTk_GradientTypeUNP&);
    dotk::DOTk_GradientTypeUNP operator=(const dotk::DOTk_GradientTypeUNP&);
};

}

#endif /* DOTK_GRADIENTTYPEUNP_HPP_ */
