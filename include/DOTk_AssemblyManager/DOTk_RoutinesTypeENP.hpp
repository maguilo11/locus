/*
 * DOTk_RoutinesTypeENP.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ROUTINESTYPEENP_HPP_
#define DOTK_ROUTINESTYPEENP_HPP_

#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

class DOTk_Primal;

template<class ScalarType>
class Vector;
template<class ScalarType>
class DOTk_ObjectiveFunction;
template<class ScalarType>
class DOTk_EqualityConstraint;

class DOTk_RoutinesTypeENP: public dotk::DOTk_AssemblyManager
{
public:
    DOTk_RoutinesTypeENP(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                         const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> >& objective_,
                         const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> >& equality_);
    virtual ~DOTk_RoutinesTypeENP();

    Real objective(const std::shared_ptr<dotk::Vector<Real> > & primal_);
    void equalityConstraint(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                            const std::shared_ptr<dotk::Vector<Real> > & output_);
    void gradient(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                  const std::shared_ptr<dotk::Vector<Real> > & dual_,
                  const std::shared_ptr<dotk::Vector<Real> > & output_);
    void jacobian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                  const std::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                  const std::shared_ptr<dotk::Vector<Real> > & output_);
    void adjointJacobian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                         const std::shared_ptr<dotk::Vector<Real> > & dual_,
                         const std::shared_ptr<dotk::Vector<Real> > & output_);
    void hessian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                 const std::shared_ptr<dotk::Vector<Real> > & dual_,
                 const std::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                 const std::shared_ptr<dotk::Vector<Real> > & output_);

private:
    void initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void allocate(dotk::types::variable_t type_, const std::shared_ptr<dotk::Vector<Real> > & data_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_StateWorkVector;
    std::shared_ptr<dotk::Vector<Real> > m_ControlWorkVector;

    std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;
    std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > m_EqualityConstraint;

private:
    DOTk_RoutinesTypeENP(const dotk::DOTk_RoutinesTypeENP &);
    dotk::DOTk_RoutinesTypeENP & operator=(const dotk::DOTk_RoutinesTypeENP &);
};

}

#endif /* DOTK_ROUTINESTYPEENP_HPP_ */
