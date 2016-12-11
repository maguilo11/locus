/*
 * DOTk_RoutinesTypeNP.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ROUTINESTYPENP_HPP_
#define DOTK_ROUTINESTYPENP_HPP_

#include <vector>
#include <tr1/memory>

#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

class DOTk_Primal;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class DOTk_ObjectiveFunction;
template<typename ScalarType>
class DOTk_EqualityConstraint;
template<typename ScalarType>
class DOTk_InequalityConstraint;

class DOTk_RoutinesTypeNP : public dotk::DOTk_AssemblyManager
{
    // TypeNP = Nonlinear Programming Assembly Manager
public:
    DOTk_RoutinesTypeNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                        const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                        const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
                        const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_);
    virtual ~DOTk_RoutinesTypeNP();

    virtual Real objective(const std::tr1::shared_ptr<dotk::Vector<Real> > & control_);
    virtual void gradient(const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & gradient_);
    virtual Real inequalityBound(const size_t index_);
    virtual Real inequalityValue(const size_t index_, const std::tr1::shared_ptr<dotk::Vector<Real> > & control_);
    virtual void inequalityGradient(const size_t index_,
                                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                                    const std::tr1::shared_ptr<dotk::Vector<Real> > & gradient_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_State;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_StateWorkVec;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_ControlWorkVec;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_EqualityConstraintDual;

    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;
    std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > m_EqualityConstraint;
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > m_InequalityConstraint;

private:
    DOTk_RoutinesTypeNP(const dotk::DOTk_RoutinesTypeNP &);
    dotk::DOTk_RoutinesTypeNP & operator=(const dotk::DOTk_RoutinesTypeNP &);
};

}

#endif /* DOTK_ROUTINESTYPENP_HPP_ */
