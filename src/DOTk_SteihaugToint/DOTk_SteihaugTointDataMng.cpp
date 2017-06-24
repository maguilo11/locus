/*
 * DOTk_SteihaugTointDataMng.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>
#include <sstream>

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

DOTk_SteihaugTointDataMng::DOTk_SteihaugTointDataMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_) :
        dotk::DOTk_OptimizationDataMng(primal_),
        m_PrimalStruc(primal_),
        m_Gradient(),
        m_AssemblyMng(std::make_shared<dotk::DOTk_RoutinesTypeULP>(objective_))
{
    this->initialize();
}

DOTk_SteihaugTointDataMng::DOTk_SteihaugTointDataMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                                     const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_OptimizationDataMng(primal_),
        m_PrimalStruc(primal_),
        m_Gradient(),
        m_AssemblyMng(std::make_shared<dotk::DOTk_RoutinesTypeUNP>(primal_, objective_, equality_))
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

void DOTk_SteihaugTointDataMng::setForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildForwardFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::FORWARD_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_SteihaugTointDataMng::setCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildCentralFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::CENTRAL_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_SteihaugTointDataMng::setBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildBackwardFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::BACKWARD_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_SteihaugTointDataMng::setParallelForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelForwardFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::PARALLEL_FORWARD_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_SteihaugTointDataMng::setParallelCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelCentralFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::PARALLEL_CENTRAL_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_SteihaugTointDataMng::setParallelBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelBackwardFiniteDiffGradient(this->getNewGradient(), m_Gradient);
    assert(m_Gradient->type() == dotk::types::PARALLEL_BACKWARD_DIFF_GRAD);
    this->setFiniteDiffPerturbationVector(input_);
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

Real DOTk_SteihaugTointDataMng::evaluateObjective(const std::shared_ptr<dotk::Vector<Real> > & input_)
{
    Real value = m_AssemblyMng->objective(input_);
    return (value);
}

size_t DOTk_SteihaugTointDataMng::getObjectiveFunctionEvaluationCounter() const
{
    return (m_AssemblyMng->getObjectiveFunctionEvaluationCounter());
}

const std::shared_ptr<dotk::DOTk_Primal> & DOTk_SteihaugTointDataMng::getPrimalStruc() const
{
    return (m_PrimalStruc);
}

const std::shared_ptr<dotk::DOTk_AssemblyManager> & DOTk_SteihaugTointDataMng::getRoutinesMng() const
{
    return (m_AssemblyMng);
}

void DOTk_SteihaugTointDataMng::initialize()
{
    this->setUserDefinedGradient();
}

void DOTk_SteihaugTointDataMng::setFiniteDiffPerturbationVector(const dotk::Vector<Real> & input_)
{
    if(input_.size() > 0)
    {
        m_Gradient->setFiniteDiffPerturbationVec(input_);
    }
    else
    {
        std::ostringstream msg;
        msg << "\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Input vector size = 0. ****\n";
        std::perror(msg.str().c_str());
        std::abort();
    }
}

}
