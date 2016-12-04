/*
 * DOTk_LineSearchQuasiNewton.cpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <fstream>
#include <iostream>

#include "vector.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_LineSearchStepMng.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_FirstOrderAlgorithm.hpp"
#include "DOTk_InverseHessianFactory.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_LineSearchQuasiNewton.hpp"
#include "DOTk_FirstOrderLineSearchAlgIO.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_LineSearchQuasiNewton::DOTk_LineSearchQuasiNewton(const std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_) :
        dotk::DOTk_FirstOrderAlgorithm(dotk::types::LINE_SEARCH_QUASI_NEWTON),
        m_InvHessianTimesVector(mng_->getNewPrimal()->clone()),
        m_IO(new dotk::DOTk_FirstOrderLineSearchAlgIO),
        m_LineSearch(step_),
        m_InvHessian(),
        m_DataMng(mng_)
{
    this->setBfgsSecantMethod();
}

DOTk_LineSearchQuasiNewton::~DOTk_LineSearchQuasiNewton()
{
}

const std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & DOTk_LineSearchQuasiNewton::getInvHessianPtr() const
{
    return (m_InvHessian);
}

void DOTk_LineSearchQuasiNewton::setLbfgsSecantMethod(size_t secant_storage_)
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildLbfgsInvHessian(secant_storage_, m_DataMng->getTrialStep(), m_InvHessian);
}

void DOTk_LineSearchQuasiNewton::setLdfpSecantMethod(size_t secant_storage_)
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildLdfpInvHessian(secant_storage_, m_DataMng->getTrialStep(), m_InvHessian);
}

void DOTk_LineSearchQuasiNewton::setLsr1SecantMethod(size_t secant_storage_)
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildLsr1InvHessian(secant_storage_, m_DataMng->getTrialStep(), m_InvHessian);
}

void DOTk_LineSearchQuasiNewton::setSr1SecantMethod()
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildSr1InvHessian(m_DataMng->getTrialStep(), m_InvHessian);
}

void DOTk_LineSearchQuasiNewton::setBfgsSecantMethod()
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildBfgsInvHessian(m_DataMng->getTrialStep(), m_InvHessian);
}

void DOTk_LineSearchQuasiNewton::setBarzilaiBorweinSecantMethod()
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildBarzilaiBorweinInvHessian(m_DataMng->getTrialStep(), m_InvHessian);
}

void DOTk_LineSearchQuasiNewton::printDiagnosticsAndSolutionEveryItr()
{
    m_IO->display(dotk::types::ITERATION);
}

void DOTk_LineSearchQuasiNewton::printDiagnosticsEveryItrAndSolutionAtTheEnd()
{
    m_IO->display(dotk::types::FINAL);
}

void DOTk_LineSearchQuasiNewton::getMin()
{
    m_IO->license();
    this->checkInvHessianPtr();
    m_IO->openFile("DOTk_LineSearchQuasiNewtonDiagnostics.out");

    this->initialize();

    size_t itr = 0;
    m_IO->printDiagnosticsReport(m_LineSearch, m_DataMng);
    while(itr < dotk::DOTk_FirstOrderAlgorithm::getMaxNumItr())
    {
        ++itr;
        dotk::DOTk_FirstOrderAlgorithm::setNumItrDone(itr);
        m_DataMng->setNumOptimizationItrDone(itr);

        m_InvHessian->apply(m_DataMng, m_DataMng->getNewGradient(), m_InvHessianTimesVector);
        m_InvHessian->setUpdateSecondOrderOperator(true);
        m_DataMng->getTrialStep()->copy(*m_InvHessianTimesVector);
        m_DataMng->getTrialStep()->scale(static_cast<Real>(-1.0));
        dotk::gtools::checkDescentDirection(m_DataMng->getNewGradient(),
                                            m_DataMng->getTrialStep(),
                                            dotk::DOTk_FirstOrderAlgorithm::getMinCosineAngleTol());

        m_DataMng->storeCurrentState();
        m_LineSearch->solveSubProblem(m_DataMng);

        m_IO->printDiagnosticsReport(m_LineSearch, m_DataMng);
        if(dotk::DOTk_FirstOrderAlgorithm::checkStoppingCriteria(m_DataMng) == true)
        {
            break;
        }
    }

    m_IO->closeFile();
    if(m_IO->display() != dotk::types::OFF)
    {
        dotk::printControl(m_DataMng->getNewPrimal());
    }
}

void DOTk_LineSearchQuasiNewton::initialize()
{
    Real initial_objective_function_value = m_DataMng->evaluateObjective();
    m_DataMng->setNewObjectiveFunctionValue(initial_objective_function_value);

    m_DataMng->computeGradient();
    Real initial_norm_gradient = m_DataMng->getNewGradient()->norm();
    m_DataMng->setNormNewGradient(initial_norm_gradient);

    dotk::gtools::getSteepestDescent(m_DataMng->getNewGradient(), m_DataMng->getTrialStep());
}

void DOTk_LineSearchQuasiNewton::checkInvHessianPtr()
{
    if(m_InvHessian.use_count() == 0)
    {
        dotk::DOTk_InverseHessianFactory factory;
        size_t secant_storage = 4;
        factory.buildLbfgsInvHessian(secant_storage, m_DataMng->getTrialStep(), m_InvHessian);
        std::cout
                << "\nDOTk WARNING: Inverse hessian operator was not specified by user. Inverse Hessian operator was set to LBFGS "
                << "and the secant storage will be set to 4.\n" << std::flush;
    }
}

}
