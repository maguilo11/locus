/*
 * DOTk_InexactNewtonAlgorithms.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"
#include "DOTk_FixedCriterion.hpp"
#include "DOTk_RelativeCriterion.hpp"
#include "DOTk_InexactNewtonAlgorithms.hpp"

namespace dotk
{

DOTk_InexactNewtonAlgorithms::DOTk_InexactNewtonAlgorithms(dotk::types::algorithm_t type_) :
        m_Criterion(new dotk::DOTk_RelativeCriterion(1e-2)),
        m_MaxNumOptItr(5000),
        m_NumOptItrDone(0),
        m_FvalTol(5.e-12),
        m_GradTol(1.e-8),
        m_StepTol(1.e-12),
        m_MinCosineAngleTol(1e-2),
        m_AlgorithmType(type_),
        m_StoppingCriterion(dotk::types::OPT_ALG_HAS_NOT_CONVERGED)
{
}

DOTk_InexactNewtonAlgorithms::~DOTk_InexactNewtonAlgorithms()
{
}

void DOTk_InexactNewtonAlgorithms::setRelativeTolerance(Real tolerance_)
{
    m_Criterion->set(dotk::types::RELATIVE_TOLERANCE, tolerance_);
}

void DOTk_InexactNewtonAlgorithms::setMaxNumItr(size_t itr_)
{
    m_MaxNumOptItr = itr_;
}

size_t DOTk_InexactNewtonAlgorithms::getMaxNumItr() const
{
    return (m_MaxNumOptItr);
}

void DOTk_InexactNewtonAlgorithms::setNumItrDone(size_t itr_)
{
    m_NumOptItrDone = itr_;
}

size_t DOTk_InexactNewtonAlgorithms::getNumItrDone() const
{
    return (m_NumOptItrDone);
}

void DOTk_InexactNewtonAlgorithms::setObjectiveFuncTol(Real tol_)
{
    m_FvalTol = tol_;
}

Real DOTk_InexactNewtonAlgorithms::getObjectiveFuncTol() const
{
    return (m_FvalTol);
}

void DOTk_InexactNewtonAlgorithms::setGradientTol(Real tol_)
{
    m_GradTol = tol_;
}

Real DOTk_InexactNewtonAlgorithms::getGradientTol() const
{
    return (m_GradTol);
}

void DOTk_InexactNewtonAlgorithms::setTrialStepTol(Real tol_)
{
    m_StepTol = tol_;
}

Real DOTk_InexactNewtonAlgorithms::getTrialStepTol() const
{
    return (m_StepTol);
}

void DOTk_InexactNewtonAlgorithms::setMinCosineAngleTol(Real tol_)
{
    m_MinCosineAngleTol = tol_;
}

Real DOTk_InexactNewtonAlgorithms::getMinCosineAngleTol() const
{
    return (m_MinCosineAngleTol);
}

dotk::types::algorithm_t DOTk_InexactNewtonAlgorithms::type() const
{
    return (m_AlgorithmType);
}

void DOTk_InexactNewtonAlgorithms::setStoppingCriterion(dotk::types::stop_criterion_t flag_)
{
    m_StoppingCriterion = flag_;
}

dotk::types::stop_criterion_t DOTk_InexactNewtonAlgorithms::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

void DOTk_InexactNewtonAlgorithms::setFixedStoppingCriterion(Real fixed_tolerance_)
{
    m_Criterion.reset(new dotk::DOTk_FixedCriterion(fixed_tolerance_));
}

void DOTk_InexactNewtonAlgorithms::setRelativeStoppingCriterion(Real relative_tolerance_)
{
    m_Criterion.reset(new dotk::DOTk_RelativeCriterion(relative_tolerance_));
}

void DOTk_InexactNewtonAlgorithms::setTrialStep(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                                                const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    switch(solver_->getSolverStopCriterion())
    {
        case dotk::types::NaN_CURVATURE_DETECTED:
        case dotk::types::ZERO_CURVATURE_DETECTED:
        case dotk::types::NEGATIVE_CURVATURE_DETECTED:
        case dotk::types::INF_CURVATURE_DETECTED:
        {
            dotk::gtools::getSteepestDescent(mng_->getNewGradient(), mng_->getTrialStep());
            break;
        }
        case dotk::types::TRUST_REGION_VIOLATED:
        case dotk::types::SOLVER_TOLERANCE_SATISFIED:
        case dotk::types::MAX_SOLVER_ITR_REACHED:
        case dotk::types::SOLVER_DID_NOT_CONVERGED:
        case dotk::types::NaN_RESIDUAL_NORM:
        case dotk::types::INF_RESIDUAL_NORM:
        case dotk::types::INVALID_INEXACTNESS_MEASURE:
        case dotk::types::INVALID_ORTHOGONALITY_MEASURE:
        {
            mng_->getTrialStep()->copy(*solver_->getDataMng()->getSolution());
            break;
        }
    }
}

bool DOTk_InexactNewtonAlgorithms::checkStoppingCriteria(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    /// Check reduced space algorithm convergence.\n
    /// Input: \n
    ///    mng_ = shared pointer to gradient based class data manager. \n
    ///      (const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng>)\n
    ///
    bool converged = false;
    if(this->getNumItrDone() < 1)
    {
        return (converged);
    }
    Real grad_norm = mng_->getNormNewGradient();
    Real trial_step_norm = mng_->getNormTrialStep();
    if(std::isnan(trial_step_norm))
    {
        converged = true;
        this->resetCurrentStateToFormer(mng_);
        this->setStoppingCriterion(dotk::types::NaN_TRIAL_STEP_NORM);
    }
    else if(std::isnan(grad_norm))
    {
        converged = true;
        this->resetCurrentStateToFormer(mng_);
        this->setStoppingCriterion(dotk::types::NaN_GRADIENT_NORM);
    }
    else if(trial_step_norm < this->getTrialStepTol())
    {
        converged = true;
        this->setStoppingCriterion(dotk::types::TRIAL_STEP_TOL_SATISFIED);
    }
    else if(grad_norm < this->getGradientTol())
    {
        converged = true;
        this->setStoppingCriterion(dotk::types::GRADIENT_TOL_SATISFIED);
    }
    else if(mng_->getNewObjectiveFunctionValue() < this->getObjectiveFuncTol())
    {
        converged = true;
        this->setStoppingCriterion(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED);
    }
    else if(this->getNumItrDone() >= this->getMaxNumItr())
    {
        converged = true;
        this->setStoppingCriterion(dotk::types::MAX_NUM_ITR_REACHED);
    }
    return (converged);
}

void DOTk_InexactNewtonAlgorithms::resetCurrentStateToFormer(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    mng_->getNewPrimal()->copy(*mng_->getOldPrimal());
    mng_->getNewGradient()->copy(*mng_->getOldGradient());
    mng_->setNewObjectiveFunctionValue(mng_->getOldObjectiveFunctionValue());
}

}
