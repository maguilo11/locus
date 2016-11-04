/*
 * DOTk_RoutinesTypeULP.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ROUTINESTYPEULP_HPP_
#define DOTK_ROUTINESTYPEULP_HPP_

#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

template<class Type>
class vector;
template<class Type>
class DOTk_ObjectiveFunction;

class DOTk_RoutinesTypeULP: public dotk::DOTk_AssemblyManager
{
public:
    DOTk_RoutinesTypeULP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> >& objective_);
    virtual ~DOTk_RoutinesTypeULP();

    Real objective(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_);
    void objective(const std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > & primal_,
                   const std::tr1::shared_ptr<dotk::vector<Real> > & values_);
    void objective(const std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > & primal_plus_,
                   const std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > & primal_minus_,
                   const std::tr1::shared_ptr<dotk::vector<Real> > & values_plus_,
                   const std::tr1::shared_ptr<dotk::vector<Real> > & values_minus_);
    void gradient(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                  const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_);
    void hessian(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                 const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                 const std::tr1::shared_ptr<dotk::vector<Real> > & Hess_times_vector_);

private:
    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;

private:
    DOTk_RoutinesTypeULP(const dotk::DOTk_RoutinesTypeULP &);
    dotk::DOTk_RoutinesTypeULP & operator=(const dotk::DOTk_RoutinesTypeULP &);
};

}

#endif /* DOTK_ROUTINESTYPEULP_HPP_ */
