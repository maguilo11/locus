/*
 * DOTk_LineSearchStep.cpp
 *
 *  Created on: Sep 26, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_LineSearchFactory.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_LineSearchStep::DOTk_LineSearchStep(const std::shared_ptr<dotk::DOTk_Primal> & primal_) :
        m_LineSearch()
{
    dotk::DOTk_LineSearchFactory step_factory;
    step_factory.buildCubicLineSearch(primal_->control(), m_LineSearch);
}

DOTk_LineSearchStep::~DOTk_LineSearchStep()
{
}

void DOTk_LineSearchStep::setContractionFactor(Real input_)
{
    m_LineSearch->setContractionFactor(input_);
}

void DOTk_LineSearchStep::setMaxNumIterations(size_t input_)
{
    m_LineSearch->setMaxNumLineSearchItr(input_);
}

void DOTk_LineSearchStep::setStagnationTolerance(Real input_)
{
    m_LineSearch->setStepStagnationTol(input_);
}

void DOTk_LineSearchStep::setArmijoLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                              Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildArmijoLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
}

void DOTk_LineSearchStep::setGoldsteinLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                 Real constant_,
                                                 Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildGoldsteinLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
    m_LineSearch->setConstant(constant_);
}

void DOTk_LineSearchStep::setCubicLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                             Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildCubicLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
}

void DOTk_LineSearchStep::setGoldenSectionLineSearch(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildGoldenSectionLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
}

Real DOTk_LineSearchStep::step() const
{
    return (m_LineSearch->getStepSize());
}

size_t DOTk_LineSearchStep::iterations() const
{
    return (m_LineSearch->getNumLineSearchItrDone());
}

void DOTk_LineSearchStep::build(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                dotk::types::line_search_t type_)
{
    dotk::DOTk_LineSearchFactory factory(type_);
    factory.build(primal_->control(), m_LineSearch);
}

void DOTk_LineSearchStep::solveSubProblem(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    m_LineSearch->step(mng_);

    Real new_objective_func_value = m_LineSearch->getNewObjectiveFunctionValue();
    mng_->setNewObjectiveFunctionValue(new_objective_func_value);

    mng_->computeGradient();
    Real norm_new_gradient = mng_->getNewGradient()->norm();
    mng_->setNormNewGradient(norm_new_gradient);

    Real norm_trial_step = mng_->getTrialStep()->norm();
    mng_->setNormTrialStep(norm_trial_step);
}

}
