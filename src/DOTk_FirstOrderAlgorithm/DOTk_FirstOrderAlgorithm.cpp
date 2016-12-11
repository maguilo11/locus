/*
 * DOTk_FirstOrderAlgorithm.cpp
 *
 *  Created on: Sep 18, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_FirstOrderAlgorithm.hpp"

namespace dotk
{

DOTk_FirstOrderAlgorithm::DOTk_FirstOrderAlgorithm() :
        mMaxNumItr(5000),
        mNumItrDone(0),
        mFvalTol(5.e-12),
        mGradTol(1.e-8),
        mStepTol(1.e-12),
        mMinCosineAngleTol(1e-2),
        mAlgorithmType(dotk::types::DOTk_ALGORITHM_DISABLED),
        m_StoppingCriterion(dotk::types::OPT_ALG_HAS_NOT_CONVERGED)
{
}

DOTk_FirstOrderAlgorithm::DOTk_FirstOrderAlgorithm(dotk::types::algorithm_t type_) :
        mMaxNumItr(5000),
        mNumItrDone(0),
        mFvalTol(5.e-12),
        mGradTol(1.e-8),
        mStepTol(1.e-12),
        mMinCosineAngleTol(1e-2),
        mAlgorithmType(type_),
        m_StoppingCriterion(dotk::types::OPT_ALG_HAS_NOT_CONVERGED)
{
}

DOTk_FirstOrderAlgorithm::~DOTk_FirstOrderAlgorithm()
{
}

void DOTk_FirstOrderAlgorithm::setMaxNumItr(size_t itr_)
{
    mMaxNumItr = itr_;
}

size_t DOTk_FirstOrderAlgorithm::getMaxNumItr() const
{
    return (mMaxNumItr);
}

void DOTk_FirstOrderAlgorithm::setNumItrDone(size_t itr_)
{
    mNumItrDone = itr_;
}

size_t DOTk_FirstOrderAlgorithm::getNumItrDone() const
{
    return (mNumItrDone);
}

void DOTk_FirstOrderAlgorithm::setObjectiveFuncTol(Real tol_)
{
    mFvalTol = tol_;
}

Real DOTk_FirstOrderAlgorithm::getObjectiveFuncTol() const
{
    return (mFvalTol);
}

void DOTk_FirstOrderAlgorithm::setGradientTol(Real tol_)
{
    mGradTol = tol_;
}

Real DOTk_FirstOrderAlgorithm::getGradientTol() const
{
    return (mGradTol);
}

void DOTk_FirstOrderAlgorithm::setTrialStepTol(Real tol_)
{
    mStepTol = tol_;
}

Real DOTk_FirstOrderAlgorithm::getTrialStepTol() const
{
    return (mStepTol);
}

void DOTk_FirstOrderAlgorithm::setMinCosineAngleTol(Real tol_)
{
    mMinCosineAngleTol = tol_;
}

Real DOTk_FirstOrderAlgorithm::getMinCosineAngleTol() const
{
    return (mMinCosineAngleTol);
}

void DOTk_FirstOrderAlgorithm::setAlgorithmType(dotk::types::algorithm_t type_)
{
    mAlgorithmType = type_;
}

dotk::types::algorithm_t DOTk_FirstOrderAlgorithm::getAlgorithmType() const
{
    return (mAlgorithmType);
}

void DOTk_FirstOrderAlgorithm::setStoppingCriterion(dotk::types::stop_criterion_t flag_)
{
    m_StoppingCriterion = flag_;
}

dotk::types::stop_criterion_t DOTk_FirstOrderAlgorithm::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

bool DOTk_FirstOrderAlgorithm::checkStoppingCriteria(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    /// Check reduced space algorithm convergence.\n
    /// Input: \n
    ///    mng_ = shared pointer to gradient based class data manager. \n
    ///      (const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng>)\n
    ///
    bool stop = false;
    if(this->getNumItrDone() < 1)
    {
        return (stop);
    }
    Real norm_trial_step = mng_->getNormTrialStep();
    Real norm_new_gradient = mng_->getNormNewGradient();

    if(std::isnan(norm_trial_step))
    {
        stop = true;
        this->resetCurrentStateToFormer(mng_);
        this->setStoppingCriterion(dotk::types::NaN_TRIAL_STEP_NORM);
    }
    else if(std::isnan(norm_new_gradient))
    {
        stop = true;
        this->resetCurrentStateToFormer(mng_);
        this->setStoppingCriterion(dotk::types::NaN_GRADIENT_NORM);
    }
    else if(norm_trial_step < this->getTrialStepTol())
    {
        stop = true;
        this->setStoppingCriterion(dotk::types::TRIAL_STEP_TOL_SATISFIED);
    }
    else if(norm_new_gradient < this->getGradientTol())
    {
        stop = true;
        this->setStoppingCriterion(dotk::types::GRADIENT_TOL_SATISFIED);
    }
    else if(mng_->getNewObjectiveFunctionValue() < this->getObjectiveFuncTol())
    {
        stop = true;
        this->setStoppingCriterion(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED);
    }
    else if(this->getNumItrDone() >= this->getMaxNumItr())
    {
        stop = true;
        this->setStoppingCriterion(dotk::types::MAX_NUM_ITR_REACHED);
    }
    return (stop);
}

void DOTk_FirstOrderAlgorithm::resetCurrentStateToFormer(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    /// Reset current state to former state solution \n
    /// Input/Output: \n
    ///    vector_space_ = shared pointer to reduced space algorithm vector space. \n
    ///      (const std::tr1::shared_ptr<dotk::DOTk_VectorSpaceReduced>)\n
    ///
    mng_->getNewPrimal()->update(1., *mng_->getOldPrimal(), 0.);
    mng_->getNewGradient()->update(1., *mng_->getOldGradient(), 0.);
    mng_->setNewObjectiveFunctionValue(mng_->getOldObjectiveFunctionValue());
}

}
