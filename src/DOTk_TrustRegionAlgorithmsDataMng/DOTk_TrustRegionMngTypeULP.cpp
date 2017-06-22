/*
 * DOTk_TrustRegionMngTypeULP.cpp
 *
 *  Created on: Nov 26, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

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

DOTk_TrustRegionMngTypeULP::DOTk_TrustRegionMngTypeULP(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_) :
        dotk::DOTk_TrustRegionAlgorithmsDataMng(primal_)
{
    m_RoutinesMng.reset(new dotk::DOTk_RoutinesTypeULP(objective_));
}

DOTk_TrustRegionMngTypeULP::~DOTk_TrustRegionMngTypeULP()
{
}

void DOTk_TrustRegionMngTypeULP::setForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeULP::setCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeULP::setBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeULP::setParallelForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeULP::setParallelCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeULP::setParallelBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_TrustRegionMngTypeULP::setFiniteDiffPerturbationVector(const dotk::Vector<Real> & input_)
{
    if(input_.size() == this->getNewGradient()->size())
    {
        m_FirstOrderOperator->setFiniteDiffPerturbationVec(input_);
    }
    else
    {
        std::ostringstream msg;
        msg << "\n**** ERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Input vector has incorrect dimension. ****\n";
        std::perror(msg.str().c_str());
        std::abort();
    }
}

}
