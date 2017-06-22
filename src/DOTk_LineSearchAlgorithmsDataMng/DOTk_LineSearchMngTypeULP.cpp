/*
 * DOTk_LineSearchMngTypeULP.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_RoutinesTypeULP.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_FirstOrderOperatorFactory.hpp"

namespace dotk
{

DOTk_LineSearchMngTypeULP::DOTk_LineSearchMngTypeULP(const std::shared_ptr<dotk::DOTk_Primal> & primal_) :
        dotk::DOTk_LineSearchAlgorithmsDataMng(primal_)
{
}

DOTk_LineSearchMngTypeULP::DOTk_LineSearchMngTypeULP(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_) :
        dotk::DOTk_LineSearchAlgorithmsDataMng(primal_)
{
    m_RoutinesMng.reset(new dotk::DOTk_RoutinesTypeULP(operators_));
}

DOTk_LineSearchMngTypeULP::~DOTk_LineSearchMngTypeULP()
{
}

void DOTk_LineSearchMngTypeULP::setForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeULP::setCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeULP::setBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeULP::setParallelForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeULP::setParallelCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeULP::setParallelBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeULP::setFiniteDiffPerturbationVector(const dotk::Vector<Real> & input_)
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
