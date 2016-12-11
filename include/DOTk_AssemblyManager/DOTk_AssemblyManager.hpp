/*
 * DOTk_AssemblyManager.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ASSEMBLYMANAGER_HPP_
#define DOTK_ASSEMBLYMANAGER_HPP_

#include <vector>
#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_AssemblyManager
{
public:
    DOTk_AssemblyManager();
    virtual ~DOTk_AssemblyManager();

    void resetCounters();
    size_t getHessianEvaluationCounter() const;
    void updateHessianEvaluationCounter();
    size_t getGradientEvaluationCounter() const;
    void updateGradientEvaluationCounter();
    size_t getJacobianEvaluationCounter() const;
    void updateJacobianEvaluationCounter();
    size_t getInverseJacobianStateCounter() const;
    void updateInverseJacobianStateCounter();
    size_t getAdjointJacobianEvaluationCounter() const;
    void updateAdjointJacobianEvaluationCounter();
    size_t getObjectiveFunctionEvaluationCounter() const;
    void updateObjectiveFunctionEvaluationCounter();
    size_t getEqualityConstraintEvaluationCounter() const;
    void updateEqualityConstraintEvaluationCounter();
    size_t getInequalityConstraintGradientCounter() const;
    void updateInequalityConstraintGradientCounter();
    size_t getAdjointInverseJacobianStateCounter() const;
    void updateAdjointInverseJacobianStateCounter();

    virtual Real objective(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_);
    virtual void objective(const std::vector<std::tr1::shared_ptr<dotk::Vector<Real> > > & primal_,
                           const std::tr1::shared_ptr<dotk::Vector<Real> > & output_);
    virtual void objective(const std::vector<std::tr1::shared_ptr<dotk::Vector<Real> > > & primal_plus_,
                           const std::vector<std::tr1::shared_ptr<dotk::Vector<Real> > > & primal_minus_,
                           const std::tr1::shared_ptr<dotk::Vector<Real> > & fval_plus_,
                           const std::tr1::shared_ptr<dotk::Vector<Real> > & fval_minus_);

    virtual void gradient(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & gradient_);
    virtual void gradient(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & dual_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & output_);

    virtual void hessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                         const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                         const std::tr1::shared_ptr<dotk::Vector<Real> > & output_);
    virtual void hessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                         const std::tr1::shared_ptr<dotk::Vector<Real> > & dual_,
                         const std::tr1::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                         const std::tr1::shared_ptr<dotk::Vector<Real> > & output_);

    virtual void equalityConstraint(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                                    const std::tr1::shared_ptr<dotk::Vector<Real> > & output_);
    virtual void jacobian(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & delta_primal_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & output_);
    virtual void adjointJacobian(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                                 const std::tr1::shared_ptr<dotk::Vector<Real> > & dual_,
                                 const std::tr1::shared_ptr<dotk::Vector<Real> > & output_);

    virtual Real inequalityBound(const size_t index_);
    virtual Real inequalityValue(const size_t index_, const std::tr1::shared_ptr<dotk::Vector<Real> > & control_);
    virtual void inequalityGradient(const size_t index_,
                                    const std::tr1::shared_ptr<dotk::Vector<Real> > & control_,
                                    const std::tr1::shared_ptr<dotk::Vector<Real> > & gradient_);

private:
    size_t m_HessianEvaluationCounter;
    size_t m_GradientEvaluationCounter;
    size_t m_JacobianEvaluationCounter;
    size_t m_InverseJacobianStateCounter;
    size_t m_AdjointJacobianEvaluationCounter;
    size_t m_ObjectiveFunctionEvaluationCounter;
    size_t m_EqualityConstraintEvaluationCounter;
    size_t m_InequalityConstraintGradientCounter;
    size_t m_AdjointInverseJacobianStateCounter;

private:
    DOTk_AssemblyManager(const dotk::DOTk_AssemblyManager &);
    dotk::DOTk_AssemblyManager & operator=(const dotk::DOTk_AssemblyManager &);
};

}

#endif /* DOTK_ASSEMBLYMANAGER_HPP_ */
