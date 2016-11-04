/*
 * DOTk_ProjectedLineSearchStep.cpp
 *
 *  Created on: Sep 26, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_BoundConstraint.hpp"
#include "DOTk_LineSearchFactory.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_BoundConstraintFactory.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"

namespace dotk
{

DOTk_ProjectedLineSearchStep::DOTk_ProjectedLineSearchStep(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        m_LineSearch(),
        m_BoundConstraint()
{
    dotk::DOTk_BoundConstraintFactory factory;
    factory.buildProjectionAlongFeasibleDirection(primal_, m_BoundConstraint);

    dotk::DOTk_LineSearchFactory step_factory;
    step_factory.buildCubicLineSearch(primal_->control(), m_LineSearch);
}

DOTk_ProjectedLineSearchStep::DOTk_ProjectedLineSearchStep(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                           const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_) :
        m_LineSearch(step_),
        m_BoundConstraint()
{
    dotk::DOTk_BoundConstraintFactory factory;
    factory.buildProjectionAlongFeasibleDirection(primal_, m_BoundConstraint);
}

DOTk_ProjectedLineSearchStep::~DOTk_ProjectedLineSearchStep()
{
}

void DOTk_ProjectedLineSearchStep::setMaxNumFeasibleItr(size_t itr_)
{
    m_BoundConstraint->setMaxNumFeasibleItr(itr_);
}

void DOTk_ProjectedLineSearchStep::setArmijoBoundConstraintMethodStep()
{
    m_BoundConstraint->setStepType(dotk::types::bound_step_t::ARMIJO_STEP);
}

void DOTk_ProjectedLineSearchStep::setBoundConstraintMethodContractionStep(Real input_)
{
    m_BoundConstraint->setContractionStep(input_);
}

void DOTk_ProjectedLineSearchStep::setFeasibleDirectionConstraint(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    dotk::DOTk_BoundConstraintFactory factory;
    factory.buildFeasibleDirection(primal_, m_BoundConstraint);
}

void DOTk_ProjectedLineSearchStep::setProjectionAlongFeasibleDirConstraint(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    dotk::DOTk_BoundConstraintFactory factory;
    factory.buildProjectionAlongFeasibleDirection(primal_, m_BoundConstraint);
}

void DOTk_ProjectedLineSearchStep::setContractionFactor(Real input_)
{
    m_LineSearch->setContractionFactor(input_);
}

void DOTk_ProjectedLineSearchStep::setMaxNumIterations(size_t input_)
{
    m_LineSearch->setMaxNumLineSearchItr(input_);
}

void DOTk_ProjectedLineSearchStep::setStagnationTolerance(Real input_)
{
    m_LineSearch->setStepStagnationTol(input_);
}

void DOTk_ProjectedLineSearchStep::setArmijoLineSearch(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildArmijoLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
}

void DOTk_ProjectedLineSearchStep::setGoldsteinLineSearch(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                          Real constant_,
                                                          Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildGoldsteinLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
    m_LineSearch->setConstant(constant_);
}

void DOTk_ProjectedLineSearchStep::setCubicLineSearch(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                      Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildCubicLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
}

void DOTk_ProjectedLineSearchStep::setGoldenSectionLineSearch(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                              Real contraction_factor_)
{
    dotk::DOTk_LineSearchFactory factory;
    factory.buildGoldenSectionLineSearch(primal_->control(), m_LineSearch);
    m_LineSearch->setContractionFactor(contraction_factor_);
}

Real DOTk_ProjectedLineSearchStep::step() const
{
    return (m_LineSearch->getStepSize());
}

size_t DOTk_ProjectedLineSearchStep::iterations() const
{
    return (m_LineSearch->getNumLineSearchItrDone());
}

void DOTk_ProjectedLineSearchStep::build(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                         dotk::types::line_search_t type_)
{
    dotk::DOTk_LineSearchFactory factory(type_);
    factory.build(primal_->control(), m_LineSearch);
}

void DOTk_ProjectedLineSearchStep::solveSubProblem(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    m_BoundConstraint->constraint(m_LineSearch, mng_);

    mng_->computeGradient();
    Real norm_new_gradient = mng_->getNewGradient()->norm();
    mng_->setNormNewGradient(norm_new_gradient);

    Real norm_trial_step = mng_->getTrialStep()->norm();
    mng_->setNormTrialStep(norm_trial_step);
}

}
