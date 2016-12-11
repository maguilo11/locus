/*
 * DOTk_CentralDifferenceGrad.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_CentralDifferenceGrad.hpp"
#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

DOTk_CentralDifferenceGrad::DOTk_CentralDifferenceGrad(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_FirstOrderOperator(dotk::types::CENTRAL_DIFF_GRAD),
        m_FiniteDiffPerturbationVec(vector_->clone())
{
    m_FiniteDiffPerturbationVec->fill(1e-6);
}

DOTk_CentralDifferenceGrad::~DOTk_CentralDifferenceGrad()
{
}

const std::tr1::shared_ptr<dotk::Vector<Real> > & DOTk_CentralDifferenceGrad::getFiniteDiffPerturbationVec() const
{
    return (m_FiniteDiffPerturbationVec);
}

void DOTk_CentralDifferenceGrad::setFiniteDiffPerturbationVec(const dotk::Vector<Real> & input_)
{
    m_FiniteDiffPerturbationVec->copy(input_);
}

void DOTk_CentralDifferenceGrad::getGradient(const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & interface_,
                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                                             const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_)
{
    /// Central difference approximation of the gradient operator, of the form f(primal_ + b) − f(primal_ - b) / h. \n
    /// Inputs: \n
    ///        primal_ = Vector of state solution at the i-th optimization iteration. \n
    ///        (std::tr1::shared_ptr<dotk::Vector<Real> >) \n
    ///        interface_ = instance to dotk::DOTk_AssemblyManager class \n
    ///        (std::tr1::shared_ptr<dotk::DOTk_AssemblyManager>)
    /// Output: \n
    ///        gradient_ = approximation of the gradient operator at the i-th optimization iteration \n
    ///        (std::tr1::shared_ptr<dotk::Vector<Real> >)
    for(size_t index = 0; index < primal_->size(); ++index)
    {
        Real xi_original = (*primal_)[index];
        Real epsilon = (*m_FiniteDiffPerturbationVec)[index] * xi_original;
        // forward difference
        (*primal_)[index] = xi_original + static_cast<Real>(0.5) * epsilon;
        Real fval_forward = interface_->objective(primal_);
        // Set entry i to orginal value
        (*primal_)[index] = xi_original;
        // backward difference
        (*primal_)[index] = xi_original - static_cast<Real>(0.5) * epsilon;
        Real fval_backward = interface_->objective(primal_);
        // Set entry i to orginal value
        (*primal_)[index] = xi_original;
        // get gradient i-th entry central difference approximation
        (*grad_)[index] = (fval_forward - fval_backward) / epsilon;
    }
}

void DOTk_CentralDifferenceGrad::gradient(const dotk::DOTk_OptimizationDataMng * const mng_)
{
    this->getGradient(mng_->getRoutinesMng(), mng_->getNewPrimal(), mng_->getNewGradient());
}

}
