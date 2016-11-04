/*
 * DOTk_TrustRegionMngTypeUNP.cpp
 *
 *  Created on: Apr 24, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_RoutinesTypeUNP.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_TrustRegionMngTypeUNP.hpp"
#include "DOTk_FirstOrderOperatorFactory.hpp"

namespace dotk
{

DOTk_TrustRegionMngTypeUNP::DOTk_TrustRegionMngTypeUNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_TrustRegionAlgorithmsDataMng(primal_)
{
    m_RoutinesMng.reset(new dotk::DOTk_RoutinesTypeUNP(primal_, objective_, equality_));
}

DOTk_TrustRegionMngTypeUNP::~DOTk_TrustRegionMngTypeUNP()
{
}

void DOTk_TrustRegionMngTypeUNP::setForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeUNP::setCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeUNP::setBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeUNP::setParallelForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeUNP::setParallelCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeUNP::setParallelBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(epsilon_);
}

void DOTk_TrustRegionMngTypeUNP::setFiniteDiffPerturbationVector(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->control().use_count() > 0)
    {
        m_FirstOrderOperator->setFiniteDiffPerturbationVec(*primal_->control());
    }
    else
    {
        std::perror("\n**** Error in DOTk_TrustRegionMngTypeUNP::setFiniteDiffPerturbationVector. User did not define control data. ABORT. ****\n");
        std::abort();
    }
}

}
