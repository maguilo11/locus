/*
 * DOTk_ProjectedStep.cpp
 *
 *  Created on: Oct 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_ProjectedStep.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_LineSearchFactory.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_ProjectedStep::DOTk_ProjectedStep(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal) :
        m_WorkVector(aPrimal->control()->clone()),
        m_LowerBound(aPrimal->control()->clone()),
        m_UpperBound(aPrimal->control()->clone()),
        m_LineSearch(),
        m_BoundConstraint(std::make_shared<dotk::DOTk_BoundConstraints>())
{
    dotk::DOTk_LineSearchFactory step_factory;
    step_factory.buildCubicLineSearch(aPrimal->control(), m_LineSearch);
    m_LowerBound->update(1., *aPrimal->getControlLowerBound(), 0.);
    m_UpperBound->update(1., *aPrimal->getControlUpperBound(), 0.);
}

DOTk_ProjectedStep::~DOTk_ProjectedStep()
{
}

void DOTk_ProjectedStep::setArmijoLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                             Real aContractionFactor)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildArmijoLineSearch(aPrimal->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(aContractionFactor);
}

void DOTk_ProjectedStep::setGoldsteinLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                Real aConstant,
                                                Real aContractionFactor)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildGoldsteinLineSearch(aPrimal->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(aContractionFactor);
    m_LineSearch->setConstant(aConstant);
}

void DOTk_ProjectedStep::setCubicLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                            Real aContractionFactor)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildCubicLineSearch(aPrimal->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(aContractionFactor);
}

void DOTk_ProjectedStep::setGoldenSectionLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                    Real aContractionFactor)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildGoldenSectionLineSearch(aPrimal->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(aContractionFactor);
}

void DOTk_ProjectedStep::setContractionFactor(Real aInput)
{
    m_LineSearch->setContractionFactor(aInput);
}

void DOTk_ProjectedStep::setMaxNumIterations(size_t aInput)
{
    m_LineSearch->setMaxNumLineSearchItr(aInput);
}

void DOTk_ProjectedStep::setStagnationTolerance(Real aInput)
{
    m_LineSearch->setStepStagnationTol(aInput);
}

Real DOTk_ProjectedStep::step() const
{
    return (m_LineSearch->getStepSize());
}

size_t DOTk_ProjectedStep::iterations() const
{
    return (m_LineSearch->getNumLineSearchItrDone());
}

void DOTk_ProjectedStep::build(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                               dotk::types::line_search_t aType)
{
    dotk::DOTk_LineSearchFactory factory(aType);
    factory.build(aPrimal->control(), m_LineSearch);
}

void DOTk_ProjectedStep::solveSubProblem(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng)
{
    m_WorkVector->update(1., *aMng->getNewPrimal(), 0.);
    m_WorkVector->update(1., *aMng->getTrialStep(), 1.);
    m_BoundConstraint->project(*m_LowerBound, *m_UpperBound, *m_WorkVector);
    m_WorkVector->update(-1., *aMng->getNewPrimal(), 1.);
    aMng->getTrialStep()->update(1., *m_WorkVector, 0.);
    Real norm_trial_step = aMng->getTrialStep()->norm();
    aMng->setNormTrialStep(norm_trial_step);

    m_LineSearch->step(aMng);

    Real new_objective_func_value = m_LineSearch->getNewObjectiveFunctionValue();
    aMng->setNewObjectiveFunctionValue(new_objective_func_value);

    aMng->computeGradient();
    Real norm_new_gradient = aMng->getNewGradient()->norm();
    aMng->setNormNewGradient(norm_new_gradient);
}

}
