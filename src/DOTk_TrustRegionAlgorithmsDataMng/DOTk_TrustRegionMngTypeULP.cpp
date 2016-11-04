/*
 * DOTk_TrustRegionMngTypeULP.cpp
 *
 *  Created on: Nov 26, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_HessianFactory.hpp"
#include "DOTk_RoutinesTypeULP.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_TrustRegionMngTypeULP.hpp"
#include "DOTk_FirstOrderOperatorFactory.hpp"

namespace dotk
{

DOTk_TrustRegionMngTypeULP::DOTk_TrustRegionMngTypeULP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_) :
        dotk::DOTk_TrustRegionAlgorithmsDataMng(primal_)
{
    m_RoutinesMng.reset(new dotk::DOTk_RoutinesTypeULP(objective_));
}

DOTk_TrustRegionMngTypeULP::~DOTk_TrustRegionMngTypeULP()
{
}

void DOTk_TrustRegionMngTypeULP::setForwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeULP::setCentralFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeULP::setBackwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeULP::setParallelForwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeULP::setParallelCentralFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeULP::setParallelBackwardFiniteDiffGradient
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeULP::setFiniteDiffPerturbationVector(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->control().use_count() > 0)
    {
        m_FirstOrderOperator->setFiniteDiffPerturbationVec(*primal_->control());
    }
    else
    {
        std::perror("\n**** Error in DOTk_TrustRegionMngTypeULP::setFiniteDiffPerturbationVector. User did not define control data. ABORT. ****\n");
        std::abort();
    }
}

}
