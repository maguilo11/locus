/*
 * DOTk_RoutinesTypeELP.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ROUTINESTYPEELP_HPP_
#define DOTK_ROUTINESTYPEELP_HPP_

#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

class DOTk_Primal;

template<class Type>
class vector;
template<class Type>
class DOTk_ObjectiveFunction;
template<class Type>
class DOTk_EqualityConstraint;

class DOTk_RoutinesTypeELP : public dotk::DOTk_AssemblyManager
{
public:
    DOTk_RoutinesTypeELP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                         const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> >& objective_,
                         const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> >& constraint_);
    virtual ~DOTk_RoutinesTypeELP();

    Real objective(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_);
    void equalityConstraint(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > &  output_);
    void gradient(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                  const std::tr1::shared_ptr<dotk::vector<Real> > & dual_,
                  const std::tr1::shared_ptr<dotk::vector<Real> > & output_);

    void jacobian(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                  const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                  const std::tr1::shared_ptr<dotk::vector<Real> > & output_);
    void adjointJacobian(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                         const std::tr1::shared_ptr<dotk::vector<Real> > & dual_,
                         const std::tr1::shared_ptr<dotk::vector<Real> > & output_);

    void hessian(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                 const std::tr1::shared_ptr<dotk::vector<Real> > & dual_,
                 const std::tr1::shared_ptr<dotk::vector<Real> > & delta_primal_,
                 const std::tr1::shared_ptr<dotk::vector<Real> > & output_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_WorkVector;
    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;
    std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > m_EqualityConstraint;

private:
    DOTk_RoutinesTypeELP(const dotk::DOTk_RoutinesTypeELP &);
    dotk::DOTk_RoutinesTypeELP & operator=(const dotk::DOTk_RoutinesTypeELP &);
};

}

#endif /* DOTK_ROUTINESTYPEELP_HPP_ */
