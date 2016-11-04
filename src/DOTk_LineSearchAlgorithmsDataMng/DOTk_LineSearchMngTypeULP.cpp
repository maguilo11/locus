/*
 * DOTk_LineSearchMngTypeULP.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_FirstOrderOperatorFactory.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_RoutinesTypeULP.hpp"

namespace dotk
{

DOTk_LineSearchMngTypeULP::DOTk_LineSearchMngTypeULP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_LineSearchAlgorithmsDataMng(primal_)
{
}

DOTk_LineSearchMngTypeULP::DOTk_LineSearchMngTypeULP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_) :
        dotk::DOTk_LineSearchAlgorithmsDataMng(primal_)
{
    m_RoutinesMng.reset(new dotk::DOTk_RoutinesTypeULP(operators_));
}

DOTk_LineSearchMngTypeULP::~DOTk_LineSearchMngTypeULP()
{
}

void DOTk_LineSearchMngTypeULP::setForwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_LineSearchMngTypeULP::setCentralFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_LineSearchMngTypeULP::setBackwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_LineSearchMngTypeULP::setParallelForwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_LineSearchMngTypeULP::setParallelCentralFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_LineSearchMngTypeULP::setParallelBackwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_LineSearchMngTypeULP::setFiniteDiffPerturbationVector(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->control().use_count() > 0)
    {
        m_FirstOrderOperator->setFiniteDiffPerturbationVec(*primal_->control());
    }
    else
    {
        std::perror("\n**** Error in DOTk_LineSearchMngTypeULP::setFiniteDiffPerturbationVector. User did not define control data. ABORT. ****\n");
        std::abort();
    }
}

}
