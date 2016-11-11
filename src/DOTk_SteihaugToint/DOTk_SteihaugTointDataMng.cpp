/*
 * DOTk_SteihaugTointDataMng.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_SteihaugTointDataMng.hpp"

#include "DOTk_RoutinesTypeULP.hpp"
#include "DOTk_RoutinesTypeUNP.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_FirstOrderOperatorFactory.hpp"

namespace dotk
{

DOTk_SteihaugTointDataMng::DOTk_SteihaugTointDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_) :
        dotk::DOTk_OptimizationDataMng(primal_),
        m_PrimalStruc(primal_),
        m_Gradient(),
        m_AssemblyMng(new dotk::DOTk_RoutinesTypeULP(objective_))
{
    this->initialize();
}

DOTk_SteihaugTointDataMng::DOTk_SteihaugTointDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_OptimizationDataMng(primal_),
        m_PrimalStruc(primal_),
        m_Gradient(),
        m_AssemblyMng(new dotk::DOTk_RoutinesTypeUNP(primal_, objective_, equality_))
{
    this->initialize();
}

DOTk_SteihaugTointDataMng::~DOTk_SteihaugTointDataMng()
{
}

void DOTk_SteihaugTointDataMng::setUserDefinedGradient()
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildUserDefinedGradient(m_Gradient);
    assert(m_Gradient->type() == dotk::types::USER_DEFINED_GRAD);
}

void DOTk_SteihaugTointDataMng::setForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildForwardFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::FORWARD_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_SteihaugTointDataMng::setCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildCentralFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::CENTRAL_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_SteihaugTointDataMng::setBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildBackwardFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::BACKWARD_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_SteihaugTointDataMng::setParallelForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelForwardFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::PARALLEL_FORWARD_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_SteihaugTointDataMng::setParallelCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelCentralFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::PARALLEL_CENTRAL_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_SteihaugTointDataMng::setParallelBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelBackwardFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::PARALLEL_BACKWARD_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_SteihaugTointDataMng::computeGradient()
{
    m_Gradient->gradient(this);
}

Real DOTk_SteihaugTointDataMng::evaluateObjective()
{
    Real value = m_AssemblyMng->objective(this->getNewPrimal());
    return (value);
}

Real DOTk_SteihaugTointDataMng::evaluateObjective(const std::tr1::shared_ptr<dotk::vector<Real> > & input_)
{
    Real value = m_AssemblyMng->objective(input_);
    return (value);
}

size_t DOTk_SteihaugTointDataMng::getObjectiveFunctionEvaluationCounter() const
{
    return (m_AssemblyMng->getObjectiveFunctionEvaluationCounter());
}

const std::tr1::shared_ptr<dotk::DOTk_Primal> & DOTk_SteihaugTointDataMng::getPrimalStruc() const
{
    return (m_PrimalStruc);
}

const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & DOTk_SteihaugTointDataMng::getRoutinesMng() const
{
    return (m_AssemblyMng);
}

void DOTk_SteihaugTointDataMng::initialize()
{
    this->setUserDefinedGradient();
}

void DOTk_SteihaugTointDataMng::setFiniteDiffPerturbationVector(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    if(epsilon_->control().use_count() > 0)
    {
        m_Gradient->setFiniteDiffPerturbationVec(*epsilon_->control());
    }
    else
    {
        std::perror("\n**** Error in DOTk_SteihaugTointDataMng::setFiniteDiffPerturbationVector. User did not define control data. ABORT. ****\n");
        std::abort();
    }
}

}