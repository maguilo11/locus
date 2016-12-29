/*
 * DOTk_LineSearchMngTypeUNP.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_RoutinesTypeUNP.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_FirstOrderOperatorFactory.hpp"

namespace dotk
{

DOTk_LineSearchMngTypeUNP::DOTk_LineSearchMngTypeUNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_LineSearchAlgorithmsDataMng(primal_)
{
    m_RoutinesMng.reset(new dotk::DOTk_RoutinesTypeUNP(primal_, objective_, equality_));
}

DOTk_LineSearchMngTypeUNP::~DOTk_LineSearchMngTypeUNP()
{
}

void DOTk_LineSearchMngTypeUNP::setForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeUNP::setCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeUNP::setBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeUNP::setParallelForwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelForwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeUNP::setParallelCentralFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelCentralFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeUNP::setParallelBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_)
{
    dotk::DOTk_FirstOrderOperatorFactory factory;
    factory.buildParallelBackwardFiniteDiffGradient(this->getNewGradient(), m_FirstOrderOperator);
    this->setFiniteDiffPerturbationVector(input_);
}

void DOTk_LineSearchMngTypeUNP::setFiniteDiffPerturbationVector(const dotk::Vector<Real> & input_)
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
