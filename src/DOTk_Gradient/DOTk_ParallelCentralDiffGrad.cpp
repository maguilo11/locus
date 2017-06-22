/*
 * DOTk_ParallelCentralDiffGrad.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_ParallelCentralDiffGrad.hpp"
#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

DOTk_ParallelCentralDiffGrad::DOTk_ParallelCentralDiffGrad(const std::shared_ptr<dotk::Vector<Real> > & vector_) :
        dotk::DOTk_FirstOrderOperator(dotk::types::PARALLEL_CENTRAL_DIFF_GRAD),
        m_FvalPlusEntries(vector_->clone()),
        m_FvalMinusEntries(vector_->clone()),
        m_FiniteDiffPerturbationVec(vector_->clone()),
        m_PerturbedPrimalPlusEntries(),
        m_PerturbedPrimalMinusEntries()
{
    this->initialize(vector_);
}

DOTk_ParallelCentralDiffGrad::~DOTk_ParallelCentralDiffGrad()
{
}

const std::shared_ptr<dotk::Vector<Real> > & DOTk_ParallelCentralDiffGrad::getFiniteDiffPerturbationVec() const
{
    return (m_FiniteDiffPerturbationVec);
}

void DOTk_ParallelCentralDiffGrad::setFiniteDiffPerturbationVec(const dotk::Vector<Real> & input_)
{
    m_FiniteDiffPerturbationVec->update(1., input_, 0.);
}

void DOTk_ParallelCentralDiffGrad::getGradient(Real fval_,
                                               const std::shared_ptr<dotk::DOTk_AssemblyManager> & interface_,
                                               const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                               const std::shared_ptr<dotk::Vector<Real> > & gradient_)
{
    /// Forward difference approximation of the gradient operator, of the form f(x + b) − f(x) / h. \n
    /// Inputs: \n
    ///        fval_ = Objective function value at primal_, fval_ = objective(primal_). \n
    ///        (Real). \n
    ///        interface_ = instance to dotk::DOTk_AssemblyManager class \n
    ///        (std::shared_ptr<dotk::DOTk_AssemblyManager>)
    ///        primal_ = Vector of state solution at the i-th optimization iteration. \n
    ///        (std::shared_ptr<dotk::Vector<Real> >) \n
    /// Output: \n
    ///        gradient_ = backward difference approximation of the gradient operator at the i-th optimization iteration \n
    ///        (std::shared_ptr<dotk::Vector<Real> >)

    // Perturb primal values
    for(size_t index = 0; index < primal_->size(); ++index)
    {
        m_PerturbedPrimalPlusEntries[index]->update(1., *primal_, 0.);
        m_PerturbedPrimalMinusEntries[index]->update(1., *primal_, 0.);
        Real epsilon = (*m_FiniteDiffPerturbationVec)[index] * (*primal_)[index];
        (*m_PerturbedPrimalPlusEntries[index])[index] = (*primal_)[index] + (epsilon * static_cast<Real>(0.5));
        (*m_PerturbedPrimalMinusEntries[index])[index] = (*primal_)[index] - (epsilon * static_cast<Real>(0.5));
    }
    // evaluate objective function
    m_FvalPlusEntries->fill(0.);
    m_FvalMinusEntries->fill(0.);
    interface_->objective(m_PerturbedPrimalPlusEntries,
                          m_PerturbedPrimalMinusEntries,
                          m_FvalPlusEntries,
                          m_FvalMinusEntries);
    // compute backward difference approximation to gradient operator
    for(size_t index = 0; index < gradient_->size(); ++index)
    {
        Real epsilon = (*m_FiniteDiffPerturbationVec)[index] * (*primal_)[index];
        (*gradient_)[index] = ((*m_FvalPlusEntries)[index] - (*m_FvalMinusEntries)[index]) / epsilon;
    }
}

void DOTk_ParallelCentralDiffGrad::gradient(const dotk::DOTk_OptimizationDataMng * const mng_)
{
    this->getGradient(mng_->getNewObjectiveFunctionValue(),
                      mng_->getRoutinesMng(),
                      mng_->getNewPrimal(),
                      mng_->getNewGradient());
}

void DOTk_ParallelCentralDiffGrad::initialize(const std::shared_ptr<dotk::Vector<Real> > & vector_)
{
    m_FiniteDiffPerturbationVec->fill(1e-6);

    m_PerturbedPrimalPlusEntries.reserve(vector_->size());
    m_PerturbedPrimalMinusEntries.reserve(vector_->size());
    for(size_t index = 0; index < vector_->size(); ++ index)
    {
        m_PerturbedPrimalPlusEntries.insert(m_PerturbedPrimalPlusEntries.begin() + index, vector_->clone());
        m_PerturbedPrimalMinusEntries.insert(m_PerturbedPrimalMinusEntries.begin() + index, vector_->clone());
    }
}

}
