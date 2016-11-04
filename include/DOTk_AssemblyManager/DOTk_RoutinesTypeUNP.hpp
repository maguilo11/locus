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

template<class Type>
class vector;
template<class Type>
class DOTk_ObjectiveFunction;
template<class Type>
class DOTk_EqualityConstraint;

class DOTk_RoutinesTypeUNP: public dotk::DOTk_AssemblyManager
{
public:
    DOTk_RoutinesTypeUNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                         const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                         const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~DOTk_RoutinesTypeUNP();

    Real objective(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_);
    void gradient(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                  const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_);
    void hessian(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                 const std::tr1::shared_ptr<dotk::vector<Real> > & hessian_times_vector_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    void allocate(dotk::types::variable_t type_, const std::tr1::shared_ptr<dotk::vector<Real> > & data_);
    void computeHessianTimesVector(const dotk::vector<Real> & control_,
                                   const dotk::vector<Real> & trial_step_,
                                   dotk::vector<Real> & hessian_times_vector_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_State;
    std::tr1::shared_ptr<dotk::vector<Real> > m_Dual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaState;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaDual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_StateWorkVec;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ControlWorkVec;
    std::tr1::shared_ptr<dotk::vector<Real> > m_HessCalcWorkVec;

    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;
    std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > m_EqualityConstraint;

private:
    DOTk_RoutinesTypeUNP(const dotk::DOTk_RoutinesTypeUNP &);
    dotk::DOTk_RoutinesTypeUNP & operator=(const dotk::DOTk_RoutinesTypeUNP &);
};

}

#endif /* DOTK_ROUTINESTYPEUNP_HPP_ */
