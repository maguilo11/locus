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

DOTk_ProjectedStep::DOTk_ProjectedStep(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        m_WorkVector(primal_->control()->clone()),
        m_LowerBound(primal_->control()->clone()),
        m_UpperBound(primal_->control()->clone()),
        m_LineSearch(),
        m_BoundConstraint(new dotk::DOTk_BoundConstraints)
{
    dotk::DOTk_LineSearchFactory step_factory;
    step_factory.buildCubicLineSearch(primal_->control(), m_LineSearch);
    m_LowerBound->update(1., *primal_->getControlLowerBound(), 0.);
    m_UpperBound->update(1., *primal_->getControlUpperBound(), 0.);
}

DOTk_ProjectedStep::~DOTk_ProjectedStep()
{
}

void DOTk_ProjectedStep::setArmijoLineSearch(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                             Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildArmijoLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
}

void DOTk_ProjectedStep::setGoldsteinLineSearch(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                Real constant_,
                                                Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildGoldsteinLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
    m_LineSearch->setConstant(constant_);
}

void DOTk_ProjectedStep::setCubicLineSearch(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                            Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildCubicLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
}

void DOTk_ProjectedStep::setGoldenSectionLineSearch(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                    Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildGoldenSectionLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
}

void DOTk_ProjectedStep::setContractionFactor(Real input_)
{
    m_LineSearch->setContractionFactor(input_);
}

void DOTk_ProjectedStep::setMaxNumIterations(size_t input_)
{
    m_LineSearch->setMaxNumLineSearchItr(input_);
}

void DOTk_ProjectedStep::setStagnationTolerance(Real input_)
{
    m_LineSearch->setStepStagnationTol(input_);
}

Real DOTk_ProjectedStep::step() const
{
    return (m_LineSearch->getStepSize());
}

size_t DOTk_ProjectedStep::iterations() const
{
    return (m_LineSearch->getNumLineSearchItrDone());
}

void DOTk_ProjectedStep::build(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                               dotk::types::line_search_t type_)
{
    dotk::DOTk_LineSearchFactory factory(type_);
    factory.build(primal_->control(), m_LineSearch);
}

void DOTk_ProjectedStep::solveSubProblem(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    m_WorkVector->update(1., *mng_->getNewPrimal(), 0.);
    m_WorkVector->update(1., *mng_->getTrialStep(), 1.);
    m_BoundConstraint->project(*m_LowerBound, *m_UpperBound, *m_WorkVector);
    m_WorkVector->update(-1., *mng_->getNewPrimal(), 1.);
    mng_->getTrialStep()->update(1., *m_WorkVector, 0.);
    Real norm_trial_step = mng_->getTrialStep()->norm();
    mng_->setNormTrialStep(norm_trial_step);

    m_LineSearch->step(mng_);

    Real new_objective_func_value = m_LineSearch->getNewObjectiveFunctionValue();
    mng_->setNewObjectiveFunctionValue(new_objective_func_value);

    mng_->computeGradient();
    Real norm_new_gradient = mng_->getNewGradient()->norm();
    mng_->setNormNewGradient(norm_new_gradient);
}

}
