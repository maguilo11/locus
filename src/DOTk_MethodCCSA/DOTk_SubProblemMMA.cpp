/*
 * DOTk_SubProblemMMA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_DataMngCCSA.hpp"
#include "DOTk_SubProblemMMA.hpp"
#include "DOTk_DualSolverNLCG.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_DualObjectiveFunctionMMA.hpp"

namespace dotk
{

DOTk_SubProblemMMA::DOTk_SubProblemMMA(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_) :
        dotk::DOTk_SubProblemCCSA(dotk::ccsa::subproblem_t::MMA),
        m_ObjectiveFunctionRho(0),
        m_InequalityConstraintRho(data_mng_->m_Dual->clone()),
        m_Bounds(new dotk::DOTk_BoundConstraints),
        m_DualSolver(new dotk::DOTk_DualSolverNLCG(data_mng_->m_Primal)),
        m_DualObjectiveFunction(new dotk::DOTk_DualObjectiveFunctionMMA(data_mng_))
{
    m_InequalityConstraintRho->fill(0.);
}

DOTk_SubProblemMMA::DOTk_SubProblemMMA(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_,
                   const std::shared_ptr<dotk::DOTk_DualSolverCCSA> & dual_solver_) :
        dotk::DOTk_SubProblemCCSA(dotk::ccsa::subproblem_t::MMA),
        m_ObjectiveFunctionRho(0),
        m_InequalityConstraintRho(data_mng_->m_Dual->clone()),
        m_Bounds(new dotk::DOTk_BoundConstraints),
        m_DualSolver(dual_solver_),
        m_DualObjectiveFunction(new dotk::DOTk_DualObjectiveFunctionMMA(data_mng_))
{
    m_InequalityConstraintRho->fill(0.);
}

DOTk_SubProblemMMA::~DOTk_SubProblemMMA()
{
}

dotk::ccsa::stopping_criterion_t DOTk_SubProblemMMA::getDualSolverStoppingCriterion() const
{
    return (m_DualSolver->getStoppingCriterion());
}

void DOTk_SubProblemMMA::setDualObjectiveEpsilonParameter(Real input_)
{
    m_DualObjectiveFunction->setEpsilon(input_);
}

void DOTk_SubProblemMMA::solve(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    Real scale = dotk::DOTk_SubProblemCCSA::getDualObjectiveTrialControlBoundScaling();
    m_DualObjectiveFunction->updateMovingAsymptotes(data_mng_->m_CurrentControl, data_mng_->m_CurrentSigma);
    m_DualObjectiveFunction->updateTrialControlBounds(scale, data_mng_->m_CurrentControl, data_mng_->m_CurrentSigma);
    m_DualObjectiveFunction->setCurrentObjectiveFunctionValue(data_mng_->m_CurrentObjectiveFunctionValue);
    m_DualObjectiveFunction->setCurrentInequalityConstraintResiduals(data_mng_->m_CurrentInequalityResiduals);

    m_DualObjectiveFunction->updateObjectiveCoefficientVectors(m_ObjectiveFunctionRho,
                                                               data_mng_->m_CurrentSigma,
                                                               data_mng_->m_CurrentObjectiveGradient);
    m_DualObjectiveFunction->updateInequalityCoefficientVectors(m_InequalityConstraintRho,
                                                                data_mng_->m_CurrentSigma,
                                                                data_mng_->m_CurrentInequalityGradients);

    m_DualSolver->solve(m_DualObjectiveFunction, data_mng_->m_Dual);
    m_DualObjectiveFunction->gatherTrialControl(data_mng_->m_CurrentControl);

    m_Bounds->projectActive(*data_mng_->m_ControlLowerBound,
                            *data_mng_->m_ControlUpperBound,
                            *data_mng_->m_CurrentControl,
                            *data_mng_->m_ActiveSet);

    data_mng_->evaluateFunctionValues();

    m_DualSolver->reset();
}

}
