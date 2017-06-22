/*
 * DOTk_AssemblyManager.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cstdlib>
#include <ostream>
#include <iostream>

#include "vector.hpp"
#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

DOTk_AssemblyManager::DOTk_AssemblyManager() :
        m_HessianEvaluationCounter(0),
        m_GradientEvaluationCounter(0),
        m_JacobianEvaluationCounter(0),
        m_InverseJacobianStateCounter(0),
        m_AdjointJacobianEvaluationCounter(0),
        m_ObjectiveFunctionEvaluationCounter(0),
        m_EqualityConstraintEvaluationCounter(0),
        m_InequalityConstraintGradientCounter(0),
        m_AdjointInverseJacobianStateCounter(0)
{
}

DOTk_AssemblyManager::~DOTk_AssemblyManager()
{
}

void DOTk_AssemblyManager::resetCounters()
{
    m_HessianEvaluationCounter = 0;
    m_GradientEvaluationCounter = 0;
    m_JacobianEvaluationCounter = 0;
    m_InverseJacobianStateCounter = 0;
    m_AdjointJacobianEvaluationCounter = 0;
    m_ObjectiveFunctionEvaluationCounter = 0;
    m_EqualityConstraintEvaluationCounter = 0;
    m_InequalityConstraintGradientCounter = 0;
    m_AdjointInverseJacobianStateCounter = 0;
}

size_t DOTk_AssemblyManager::getHessianEvaluationCounter() const
{
    return (m_HessianEvaluationCounter);
}

void DOTk_AssemblyManager::updateHessianEvaluationCounter()
{
    m_HessianEvaluationCounter++;
}

size_t DOTk_AssemblyManager::getGradientEvaluationCounter() const
{
    return (m_GradientEvaluationCounter);
}

void DOTk_AssemblyManager::updateGradientEvaluationCounter()
{
    m_GradientEvaluationCounter++;
}

size_t DOTk_AssemblyManager::getJacobianEvaluationCounter() const
{
    return (m_JacobianEvaluationCounter);
}

void DOTk_AssemblyManager::updateJacobianEvaluationCounter()
{
    m_JacobianEvaluationCounter++;
}

size_t DOTk_AssemblyManager::getInverseJacobianStateCounter() const
{
    return (m_InverseJacobianStateCounter);
}

void DOTk_AssemblyManager::updateInverseJacobianStateCounter()
{
    m_InverseJacobianStateCounter++;
}

size_t DOTk_AssemblyManager::getAdjointJacobianEvaluationCounter() const
{
    return (m_AdjointJacobianEvaluationCounter);
}

void DOTk_AssemblyManager::updateAdjointJacobianEvaluationCounter()
{
    m_AdjointJacobianEvaluationCounter++;
}

size_t DOTk_AssemblyManager::getObjectiveFunctionEvaluationCounter() const
{
    return (m_ObjectiveFunctionEvaluationCounter);
}

void DOTk_AssemblyManager::updateObjectiveFunctionEvaluationCounter()
{
    m_ObjectiveFunctionEvaluationCounter++;
}

size_t DOTk_AssemblyManager::getEqualityConstraintEvaluationCounter() const
{
    return (m_EqualityConstraintEvaluationCounter);
}

void DOTk_AssemblyManager::updateEqualityConstraintEvaluationCounter()
{
    m_EqualityConstraintEvaluationCounter++;
}

size_t DOTk_AssemblyManager::getInequalityConstraintGradientCounter() const
{
    return (m_InequalityConstraintGradientCounter);
}

void DOTk_AssemblyManager::updateInequalityConstraintGradientCounter()
{
    m_InequalityConstraintGradientCounter++;
}

size_t DOTk_AssemblyManager::getAdjointInverseJacobianStateCounter() const
{
    return (m_AdjointInverseJacobianStateCounter);
}

void DOTk_AssemblyManager::updateAdjointInverseJacobianStateCounter()
{
    m_AdjointInverseJacobianStateCounter++;
}

Real DOTk_AssemblyManager::objective(const std::shared_ptr<dotk::Vector<Real> > & primal_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::objective **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::objective(const std::vector<std::shared_ptr<dotk::Vector<Real> > > & primal_,
                                     const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::objective **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::objective(const std::vector<std::shared_ptr<dotk::Vector<Real> > > & primal_plus_,
                                     const std::vector<std::shared_ptr<dotk::Vector<Real> > > & primal_minus_,
                                     const std::shared_ptr<dotk::Vector<Real> > & fval_plus_,
                                     const std::shared_ptr<dotk::Vector<Real> > & fval_minus_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::objective **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::gradient(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                    const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::gradient **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::gradient(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                    const std::shared_ptr<dotk::Vector<Real> > & dual_,
                                    const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::gradient **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::hessian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                   const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                   const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::hessian **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::hessian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                   const std::shared_ptr<dotk::Vector<Real> > & dual_,
                                   const std::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                                   const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::hessian **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::equalityConstraint(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                              const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::equalityConstraint **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::jacobian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                    const std::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                                    const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::jacobian **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::adjointJacobian(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                           const std::shared_ptr<dotk::Vector<Real> > & dual_,
                                           const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::adjointJacobian **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

Real DOTk_AssemblyManager::inequalityBound(const size_t index_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::inequalityBound **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

Real DOTk_AssemblyManager::inequalityValue(const size_t index_,
                                           const std::shared_ptr<dotk::Vector<Real> > & control_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::inequalityValue **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

void DOTk_AssemblyManager::inequalityGradient(const size_t index_,
                                              const std::shared_ptr<dotk::Vector<Real> > & control_,
                                              const std::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    std::string msg(" CALLING UNIMPLEMENTED DOTk_AssemblyManager::inequalityGradient **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    std::abort();
}

}
