/*
 * DOTk_SteihaugTointStepMng.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_Preconditioner.hpp"
#include "DOTk_SteihaugTointPcg.hpp"
#include "DOTk_SteihaugTointStepMng.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_SteihaugTointNewtonIO.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_SteihaugTointStepMng::DOTk_SteihaugTointStepMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_) :
        dotk::DOTk_TrustRegionStepMng(),
        m_CurrentPrimal(primal_->control()),
        m_LinearOperator(linear_operator_),
        m_Preconditioner(new dotk::DOTk_Preconditioner)
{
}

DOTk_SteihaugTointStepMng::DOTk_SteihaugTointStepMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_) :
        dotk::DOTk_TrustRegionStepMng(),
        m_CurrentPrimal(primal_->control()),
        m_LinearOperator(linear_operator_),
        m_Preconditioner(preconditioner_)
{
}

DOTk_SteihaugTointStepMng::~DOTk_SteihaugTointStepMng()
{
}

void DOTk_SteihaugTointStepMng::setNumOptimizationItrDone(const size_t & itr_)
{
    m_LinearOperator->setNumOtimizationItrDone(itr_);
}

void DOTk_SteihaugTointStepMng::solveSubProblem(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                                const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                                const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointNewtonIO> & io_)
{
    Real new_objective_value = 0.;
    this->setNumTrustRegionSubProblemItrDone(1);
    m_CurrentPrimal->update(1., *mng_->getNewPrimal(), 0.);
    Real angle_tolerance = this->getMinCosineAngleTolerance();
    Real current_objective_value = mng_->getNewObjectiveFunctionValue();
    Real min_trust_region_radius = dotk::DOTk_TrustRegionStepMng::getMinTrustRegionRadius();
    if(dotk::DOTk_TrustRegionStepMng::getTrustRegionRadius() < min_trust_region_radius)
    {
        dotk::DOTk_TrustRegionStepMng::setTrustRegionRadius(min_trust_region_radius);
    }

    size_t max_num_itr = this->getMaxNumTrustRegionSubProblemItr();
    while(this->getNumTrustRegionSubProblemItrDone() <= max_num_itr)
    {
        solver_->setTrustRegionRadius(this->getTrustRegionRadius());
        solver_->solve(m_Preconditioner, m_LinearOperator, mng_);
        m_LinearOperator->apply(mng_, mng_->getTrialStep(), mng_->getMatrixTimesVector());
        dotk::gtools::checkDescentDirection(mng_->getNewGradient(), mng_->getTrialStep(), angle_tolerance);

        Real gradient_dot_trial_step = mng_->getNewGradient()->dot(*mng_->getTrialStep());
        Real trial_step_dot_hess_times_trial_step = mng_->getTrialStep()->dot(*mng_->getMatrixTimesVector());
        Real predicted_reduction = (gradient_dot_trial_step + trial_step_dot_hess_times_trial_step);
        this->setPredictedReduction(predicted_reduction);

        dotk::DOTk_TrustRegionStepMng::updateAdaptiveObjectiveInexactnessTolerance();

        mng_->getNewPrimal()->update(1., *mng_->getTrialStep(), 1.);
        new_objective_value = mng_->evaluateObjective();
        Real actual_reduction = new_objective_value - current_objective_value;
        this->setActualReduction(actual_reduction);

        Real actual_over_pred_reduction = actual_reduction / predicted_reduction;
        this->setActualOverPredictedReduction(actual_over_pred_reduction);

        io_->printTrustRegionSubProblemDiagnostics(mng_, solver_, this);
        if(this->updateTrustRegionRadius() == true)
        {
            break;
        }
        this->updateNumTrustRegionSubProblemItrDone();
        mng_->getNewPrimal()->update(1., *m_CurrentPrimal, 0.);
    }

    mng_->setOldObjectiveFunctionValue(current_objective_value);
    mng_->setNewObjectiveFunctionValue(new_objective_value);
    this->updateDataManager(mng_);
    m_LinearOperator->updateLimitedMemoryStorage(true);
}

void DOTk_SteihaugTointStepMng::updateDataManager(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    mng_->getOldPrimal()->update(1., *m_CurrentPrimal, 0.);
    mng_->getOldGradient()->update(1., *mng_->getNewGradient(), 0.);
    mng_->computeGradient();

    Real norm_new_gradient = mng_->getNewGradient()->norm();
    mng_->setNormNewGradient(norm_new_gradient);

    Real norm_trial_step = mng_->getTrialStep()->norm();
    mng_->setNormTrialStep(norm_trial_step);
}

}
