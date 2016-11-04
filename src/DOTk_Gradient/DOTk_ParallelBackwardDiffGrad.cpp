/*
 * DOTk_ParallelBackwardDiffGrad.cpp
 *
 *  Created on: Feb 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_ParallelBackwardDiffGrad.hpp"
#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

DOTk_ParallelBackwardDiffGrad::DOTk_ParallelBackwardDiffGrad(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_) :
        dotk::DOTk_FirstOrderOperator(dotk::types::PARALLEL_BACKWARD_DIFF_GRAD),
        m_Fval(vector_->clone()),
        m_FiniteDiffPerturbationVec(vector_->clone()),
        m_PerturbedPrimal()
{
    this->initialize(vector_);
}

DOTk_ParallelBackwardDiffGrad::~DOTk_ParallelBackwardDiffGrad()
{
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_ParallelBackwardDiffGrad::getFiniteDiffPerturbationVec() const
{
    return (m_FiniteDiffPerturbationVec);
}

void DOTk_ParallelBackwardDiffGrad::setFiniteDiffPerturbationVec(const dotk::vector<Real> & input_)
{
    m_FiniteDiffPerturbationVec->copy(input_);
}

void DOTk_ParallelBackwardDiffGrad::getGradient(Real fval_,
                                                const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & interface_,
                                                const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                                const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_)
{
    /// Forward difference approximation of the gradient operator, of the form f(x + b) − f(x) / h. \n
    /// Inputs: \n
    ///        fval_ = Objective function value at primal_, fval_ = objective(primal_). \n
    ///        (Real). \n
    ///        interface_ = instance to dotk::DOTk_AssemblyManager class \n
    ///        (std::tr1::shared_ptr<dotk::DOTk_AssemblyManager>)
    ///        primal_ = Vector of state solution at the i-th optimization iteration. \n
    ///        (std::tr1::shared_ptr<dotk::vector<Real> >) \n
    /// Output: \n
    ///        gradient_ = backward difference approximation of the gradient operator at the i-th optimization iteration \n
    ///        (std::tr1::shared_ptr<dotk::vector<Real> >)

    // Perturb primal values
    m_Fval->fill(static_cast<Real>(0.));
    for(size_t index = 0; index < primal_->size(); ++index)
    {
        m_PerturbedPrimal[index]->copy(*primal_);
        Real original_entry = (*primal_)[index];
        Real epsilon = (*m_FiniteDiffPerturbationVec)[index] * original_entry;
        Real new_entry = (*primal_)[index] - epsilon;
        (*m_PerturbedPrimal[index])[index] = new_entry;
    }
    // evaluate objective function
    interface_->objective(m_PerturbedPrimal, m_Fval);
    // compute backward difference approximation to gradient operator
    for(size_t index = 0; index < gradient_->size(); ++index)
    {
        Real epsilon = (*m_FiniteDiffPerturbationVec)[index] * (*primal_)[index];
        (*gradient_)[index] = (fval_ - (*m_Fval)[index]) / epsilon;
    }
}

void DOTk_ParallelBackwardDiffGrad::gradient(const dotk::DOTk_OptimizationDataMng * const mng_)
{
    this->getGradient(mng_->getNewObjectiveFunctionValue(),
                      mng_->getRoutinesMng(),
                      mng_->getNewPrimal(),
                      mng_->getNewGradient());
}

void DOTk_ParallelBackwardDiffGrad::initialize(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_)
{
    m_FiniteDiffPerturbationVec->fill(1e-6);

    m_PerturbedPrimal.reserve(vector_->size());
    for(size_t index = 0; index < vector_->size(); ++ index)
    {
        m_PerturbedPrimal.insert(m_PerturbedPrimal.begin() + index, vector_->clone());
    }
}

}
