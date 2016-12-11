/*
 * DOTk_ForwardDifferenceGrad.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_ForwardDifferenceGrad.hpp"
#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

DOTk_ForwardDifferenceGrad::DOTk_ForwardDifferenceGrad(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_FirstOrderOperator(dotk::types::FORWARD_DIFF_GRAD),
        m_FiniteDiffPerturbationVec(vector_->clone())
{
    m_FiniteDiffPerturbationVec->fill(1e-6);
}

DOTk_ForwardDifferenceGrad::~DOTk_ForwardDifferenceGrad()
{
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_ForwardDifferenceGrad::getFiniteDiffPerturbationVec() const
{
    return (m_FiniteDiffPerturbationVec);
}

void DOTk_ForwardDifferenceGrad::setFiniteDiffPerturbationVec(const dotk::Vector<Real> & input_)
{
    m_FiniteDiffPerturbationVec->copy(input_);
}

void DOTk_ForwardDifferenceGrad::getGradient(Real fval_,
                                             const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & interface_,
                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_)
{
    /// Forward difference approximation of the gradient operator, of the form f(x + b) − f(x) / h. \n
    /// Inputs: \n
    ///        fval_ = Objective function value at primal_, fval_ = evaluate(primal_). \n
    ///        (Real). \n
    ///        interface_ = instance to dotk::DOTk_AssemblyManager class \n
    ///        (std::tr1::shared_ptr<dotk::DOTk_AssemblyManager>)
    ///        primal_ = Vector of state solution at the i-th optimization iteration. \n
    ///        (std::tr1::shared_ptr<dotk::Vector<Real> >) \n
    /// Output: \n
    ///        gradient_ = backward difference approximation of the gradient operator at the i-th optimization iteration \n
    ///        (std::tr1::shared_ptr<dotk::Vector<Real> >)
    for(size_t index = 0; index < primal_->size(); ++index)
    {
        // copy original state solution i-th element
        Real xi_original = (*primal_)[index];
        // compute perturbation
        Real epsilon = (*m_FiniteDiffPerturbationVec)[index] * xi_original;
        // apply forward difference to i-th element
        (*primal_)[index] = xi_original + epsilon;
        Real fval_forward = interface_->objective(primal_);
        // get i-th gradient entry
        (*grad_)[index] = (fval_forward - fval_) / epsilon;
        // Set entry i to original value
        (*primal_)[index] = xi_original;
    }
}

void DOTk_ForwardDifferenceGrad::gradient(const dotk::DOTk_OptimizationDataMng * const mng_)
{
    this->getGradient(mng_->getNewObjectiveFunctionValue(),
                      mng_->getRoutinesMng(),
                      mng_->getNewPrimal(),
                      mng_->getNewGradient());
}

}
