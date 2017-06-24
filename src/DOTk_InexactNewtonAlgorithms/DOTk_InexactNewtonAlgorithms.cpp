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

DOTk_InexactNewtonAlgorithms::DOTk_InexactNewtonAlgorithms(dotk::types::algorithm_t aType) :
        m_Criterion(std::make_shared<dotk::DOTk_RelativeCriterion>(1e-2)),
        m_MaxNumOptItr(5000),
        m_NumOptItrDone(0),
        m_FvalTol(5.e-12),
        m_GradTol(1.e-8),
        m_StepTol(1.e-12),
        m_MinCosineAngleTol(1e-2),
        m_AlgorithmType(aType),
        m_StoppingCriterion(dotk::types::OPT_ALG_HAS_NOT_CONVERGED)
{
}

DOTk_InexactNewtonAlgorithms::~DOTk_InexactNewtonAlgorithms()
{
}

void DOTk_InexactNewtonAlgorithms::setRelativeTolerance(Real aInput)
{
    m_Criterion->set(dotk::types::RELATIVE_TOLERANCE, aInput);
}

void DOTk_InexactNewtonAlgorithms::setMaxNumItr(size_t aInput)
{
    m_MaxNumOptItr = aInput;
}

size_t DOTk_InexactNewtonAlgorithms::getMaxNumItr() const
{
    return (m_MaxNumOptItr);
}

void DOTk_InexactNewtonAlgorithms::setNumItrDone(size_t aInput)
{
    m_NumOptItrDone = aInput;
}

size_t DOTk_InexactNewtonAlgorithms::getNumItrDone() const
{
    return (m_NumOptItrDone);
}

void DOTk_InexactNewtonAlgorithms::setObjectiveFuncTol(Real aInput)
{
    m_FvalTol = aInput;
}

Real DOTk_InexactNewtonAlgorithms::getObjectiveFuncTol() const
{
    return (m_FvalTol);
}

void DOTk_InexactNewtonAlgorithms::setGradientTol(Real aInput)
{
    m_GradTol = aInput;
}

Real DOTk_InexactNewtonAlgorithms::getGradientTol() const
{
    return (m_GradTol);
}

void DOTk_InexactNewtonAlgorithms::setTrialStepTol(Real aInput)
{
    m_StepTol = aInput;
}

Real DOTk_InexactNewtonAlgorithms::getTrialStepTol() const
{
    return (m_StepTol);
}

void DOTk_InexactNewtonAlgorithms::setMinCosineAngleTol(Real aInput)
{
    m_MinCosineAngleTol = aInput;
}

Real DOTk_InexactNewtonAlgorithms::getMinCosineAngleTol() const
{
    return (m_MinCosineAngleTol);
}

dotk::types::algorithm_t DOTk_InexactNewtonAlgorithms::type() const
{
    return (m_AlgorithmType);
}

void DOTk_InexactNewtonAlgorithms::setStoppingCriterion(dotk::types::stop_criterion_t aInput)
{
    m_StoppingCriterion = aInput;
}

dotk::types::stop_criterion_t DOTk_InexactNewtonAlgorithms::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

void DOTk_InexactNewtonAlgorithms::setFixedStoppingCriterion(Real aInput)
{
    m_Criterion = std::make_shared<dotk::DOTk_FixedCriterion>(aInput);
}

void DOTk_InexactNewtonAlgorithms::setRelativeStoppingCriterion(Real aInput)
{
    m_Criterion = std::make_shared<dotk::DOTk_RelativeCriterion>(aInput);
}

void DOTk_InexactNewtonAlgorithms::setTrialStep(const std::shared_ptr<dotk::DOTk_KrylovSolver> & aSolver,
                                                const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    switch(aSolver->getSolverStopCriterion())
    {
        case dotk::types::NaN_CURVATURE_DETECTED:
        case dotk::types::ZERO_CURVATURE_DETECTED:
        case dotk::types::NEGATIVE_CURVATURE_DETECTED:
        case dotk::types::INF_CURVATURE_DETECTED:
        {
            dotk::gtools::getSteepestDescent(aMng->getNewGradient(), aMng->getTrialStep());
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
            aMng->getTrialStep()->update(1., *aSolver->getDataMng()->getSolution(), 0.);
            break;
        }
    }
}

bool DOTk_InexactNewtonAlgorithms::checkStoppingCriteria(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    /// Check reduced space algorithm convergence.\n
    /// Input: \n
    ///    aMng = shared pointer to gradient based class data manager. \n
    ///      (const std::shared_ptr<dotk::DOTk_OptimizationDataMng>)\n
    ///
    bool converged = false;
    if(this->getNumItrDone() < 1)
    {
        return (converged);
    }
    Real grad_norm = aMng->getNormNewGradient();
    Real trial_step_norm = aMng->getNormTrialStep();
    if(std::isnan(trial_step_norm))
    {
        converged = true;
        this->resetCurrentStateToFormer(aMng);
        this->setStoppingCriterion(dotk::types::NaN_TRIAL_STEP_NORM);
    }
    else if(std::isnan(grad_norm))
    {
        converged = true;
        this->resetCurrentStateToFormer(aMng);
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
    else if(aMng->getNewObjectiveFunctionValue() < this->getObjectiveFuncTol())
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

void DOTk_InexactNewtonAlgorithms::resetCurrentStateToFormer(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    aMng->getNewPrimal()->update(1., *aMng->getOldPrimal(), 0.);
    aMng->getNewGradient()->update(1., *aMng->getOldGradient(), 0.);
    aMng->setNewObjectiveFunctionValue(aMng->getOldObjectiveFunctionValue());
}

}
