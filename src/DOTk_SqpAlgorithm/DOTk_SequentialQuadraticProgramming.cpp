/*
 * DOTk_SequentialQuadraticProgramming.cpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"
#include "DOTk_SequentialQuadraticProgramming.hpp"

namespace dotk
{

DOTk_SequentialQuadraticProgramming::DOTk_SequentialQuadraticProgramming(dotk::types::algorithm_t type_) :
        m_MaxNumOptItr(50),
        m_NumOptItrDone(0),
        m_NumTrustRegionSubProblemItrDone(0),
        m_GradientTolerance(1e-10),
        m_TrialStepTolerance(1e-10),
        m_OptimalityTolerance(1e-10),
        m_FeasibilityTolerance(1e-12),
        m_AlgorithmType(type_),
        m_StoppingCriterion(dotk::types::OPT_ALG_HAS_NOT_CONVERGED)
{
}

DOTk_SequentialQuadraticProgramming::~DOTk_SequentialQuadraticProgramming()
{
}

size_t DOTk_SequentialQuadraticProgramming::getMaxNumItr() const
{
    return (m_MaxNumOptItr);
}

void DOTk_SequentialQuadraticProgramming::setMaxNumItr(size_t itr_)
{
    m_MaxNumOptItr = itr_;
}

void DOTk_SequentialQuadraticProgramming::setNumItrDone(size_t itr_)
{
    m_NumOptItrDone = itr_;
}

size_t DOTk_SequentialQuadraticProgramming::getNumItrDone() const
{
    return (m_NumOptItrDone);
}

size_t DOTk_SequentialQuadraticProgramming::getNumTrustRegionSubProblemItrDone() const
{
    return (m_NumTrustRegionSubProblemItrDone);
}

void DOTk_SequentialQuadraticProgramming::setNumTrustRegionSubProblemItrDone(size_t itr_)
{
    m_NumTrustRegionSubProblemItrDone = itr_;
}

void DOTk_SequentialQuadraticProgramming::setGradientTolerance(Real tol_)
{
    m_GradientTolerance = tol_;
}

Real DOTk_SequentialQuadraticProgramming::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

Real DOTk_SequentialQuadraticProgramming::getTrialStepTolerance() const
{
    return (m_TrialStepTolerance);
}

void DOTk_SequentialQuadraticProgramming::setOptimalityTolerance(Real tol_)
{
    m_OptimalityTolerance = tol_;
}

Real DOTk_SequentialQuadraticProgramming::getOptimalityTolerance() const
{
    return (m_OptimalityTolerance);
}

void DOTk_SequentialQuadraticProgramming::setFeasibilityTolerance(Real tol_)
{
    m_FeasibilityTolerance = tol_;
}

Real DOTk_SequentialQuadraticProgramming::getFeasibilityTolerance() const
{
    return (m_FeasibilityTolerance);
}

void DOTk_SequentialQuadraticProgramming::setStoppingCriterion(dotk::types::stop_criterion_t flag_)
{
    m_StoppingCriterion = flag_;
}

dotk::types::stop_criterion_t DOTk_SequentialQuadraticProgramming::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

bool DOTk_SequentialQuadraticProgramming::checkStoppingCriteria
(const std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    size_t itr = this->getNumItrDone();
    size_t max_num_itr = this->getMaxNumItr();
    Real grad_tol = this->getGradientTolerance();
    Real trial_step_tol = this->getTrialStepTolerance();
    Real feasibility_tol = this->getFeasibilityTolerance();

    Real grad_norm = mng_->getNewGradient()->norm();
    Real trial_step_norm = mng_->getTrialStep()->norm();
    Real eq_constraint_residual_norm = mng_->getNewEqualityConstraintResidual()->norm();

    Real trust_region_radius = mng_->getTrustRegionRadius();

    bool stop = false;
    this->setStoppingCriterion(dotk::types::OPT_ALG_HAS_NOT_CONVERGED);
    if(itr >= max_num_itr)
    {
        this->setStoppingCriterion(dotk::types::MAX_NUM_ITR_REACHED);
        stop = true;
    }
    else if((grad_norm < grad_tol) && (eq_constraint_residual_norm < feasibility_tol))
    {
        this->setStoppingCriterion(dotk::types::OPTIMALITY_AND_FEASIBILITY_SATISFIED);
        stop = true;
    }
    else if(trust_region_radius < trial_step_tol)
    {
        this->setStoppingCriterion(dotk::types::TRUST_REGION_RADIUS_SMALLER_THAN_TRIAL_STEP_NORM);
        stop = true;
    }
    else if(std::isnan(trial_step_norm))
    {
        stop = true;
        this->resetCurrentStateToFormer(mng_);
        this->setStoppingCriterion(dotk::types::NaN_TRIAL_STEP_NORM);
    }
    else if(std::isnan(grad_norm))
    {
        stop = true;
        this->resetCurrentStateToFormer(mng_);
        this->setStoppingCriterion(dotk::types::NaN_GRADIENT_NORM);
    }
    return (stop);
}

void DOTk_SequentialQuadraticProgramming::storePreviousSolution
(const std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    mng_->setOldObjectiveFunctionValue(mng_->getNewObjectiveFunctionValue());
    mng_->getOldDual()->update(1., *mng_->getNewDual(), 0.);
    mng_->getOldPrimal()->update(1., *mng_->getNewPrimal(), 0.);
    mng_->getOldGradient()->update(1., *mng_->getNewGradient(), 0.);
    mng_->getOldEqualityConstraintResidual()->update(1., *mng_->getNewEqualityConstraintResidual(), 0.);
}

void DOTk_SequentialQuadraticProgramming::resetCurrentStateToFormer
(const std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    mng_->getNewPrimal()->update(1., *mng_->getOldPrimal(), 0.);
    mng_->getNewGradient()->update(1., *mng_->getOldGradient(), 0.);
    mng_->setNewObjectiveFunctionValue(mng_->getOldObjectiveFunctionValue());
}

}
