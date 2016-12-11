/*
 * DOTk_RoutinesTypeUNP.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ROUTINESTYPEUNP_HPP_
#define DOTK_ROUTINESTYPEUNP_HPP_

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

class DOTk_RoutinesTypeUNP: public dotk::DOTk_AssemblyManager
{
public:
    DOTk_RoutinesTypeUNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                         const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                         const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~DOTk_RoutinesTypeUNP();

    Real objective(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_);
    void gradient(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                  const std::tr1::shared_ptr<dotk::Vector<Real> > & gradient_);
    void hessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                 const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                 const std::tr1::shared_ptr<dotk::Vector<Real> > & hessian_times_vector_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    void allocate(dotk::types::variable_t type_, const std::tr1::shared_ptr<dotk::Vector<Real> > & data_);
    void computeHessianTimesVector(const dotk::Vector<Real> & control_,
                                   const dotk::Vector<Real> & trial_step_,
                                   dotk::Vector<Real> & hessian_times_vector_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_State;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_Dual;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaState;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_DeltaDual;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_StateWorkVec;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_ControlWorkVec;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_HessCalcWorkVec;

    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;
    std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > m_EqualityConstraint;

private:
    DOTk_RoutinesTypeUNP(const dotk::DOTk_RoutinesTypeUNP &);
    dotk::DOTk_RoutinesTypeUNP & operator=(const dotk::DOTk_RoutinesTypeUNP &);
};

}

#endif /* DOTK_ROUTINESTYPEUNP_HPP_ */
