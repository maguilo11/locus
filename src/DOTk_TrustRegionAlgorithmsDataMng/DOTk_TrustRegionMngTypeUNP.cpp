/*
 * DOTk_TrustRegionMngTypeUNP.cpp
 *
 *  Created on: Apr 24, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

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

DOTk_TrustRegionMngTypeUNP::DOTk_TrustRegionMngTypeUNP(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                                       const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_TrustRegionAlgorithmsDataMng(primal_)
{
    m_RoutinesMng = std::make_shared<dotk::DOTk_RoutinesTypeUNP>(primal_, objective_, equality_);
}

DOTk_TrustRegionMngTypeUNP::~DOTk_TrustRegionMngTypeUNP()
{
}

void DOTk_TrustRegionMngTypeUNP::setForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeUNP::setCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeUNP::setBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeUNP::setParallelForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeUNP::setParallelCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeUNP::setParallelBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeUNP::setFiniteDiffPerturbationVector(const dotk::Vector<Real> & input_)
{
    if(input_.size() == this->getNewGradient()->size())
    {
        m_FirstOrderOperator->setFiniteDiffPerturbationVec(input_);
    }
    else
    {
        std::ostringstream msg;
        msg << "\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Input vector has incorrect dimension. ****\n";
        std::perror(msg.str().c_str());
        std::abort();
    }
}

}
