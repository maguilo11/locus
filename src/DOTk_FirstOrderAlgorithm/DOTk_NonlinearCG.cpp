/*
 * DOTk_NonlinearCG.cpp
 *
 *  Created on: Sep 18, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <fstream>
#include <ostream>

#include "vector.hpp"
#include "DOTk_NonlinearCG.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_DescentDirection.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_LineSearchStepMng.hpp"
#include "DOTk_NonlinearCGFactory.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_FirstOrderLineSearchAlgIO.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_NonlinearCG::DOTk_NonlinearCG(const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & aStep,
                                   const std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & aMng) :
        dotk::DOTk_FirstOrderAlgorithm(dotk::types::NONLINEAR_CG),
        m_LineSearch(aStep),
        m_IO(std::make_shared<dotk::DOTk_FirstOrderLineSearchAlgIO>()),
        m_DescentDirection(),
        m_DataMng(aMng)
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildFletcherReevesNlcg(m_DescentDirection);
}

DOTk_NonlinearCG::~DOTk_NonlinearCG()
{
}

void DOTk_NonlinearCG::setDanielsNlcg(const std::shared_ptr<dotk::DOTk_LinearOperator> & aInput)
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildDanielsNlcg(aInput, m_DescentDirection);
}

void DOTk_NonlinearCG::setFletcherReevesNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildFletcherReevesNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::setPolakRibiereNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildPolakRibiereNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::setHestenesStiefelNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildHestenesStiefelNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::setConjugateDescentNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildConjugateDescentNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::setHagerZhangNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildHagerZhangNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::setDaiLiaoNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildDaiLiaoNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::setDaiYuanNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildDaiYuanNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::setDaiYuanHybridNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildDaiYuanHybridNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::setPerryShannoNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildPerryShannoNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::setLiuStoreyNlcg()
{
    dotk::DOTk_NonlinearCGFactory factory;
    factory.buildLiuStoreyNlcg(m_DescentDirection);
}

void DOTk_NonlinearCG::printDiagnosticsAndSolutionEveryItr()
{
    m_IO->display(dotk::types::ITERATION);
}

void DOTk_NonlinearCG::printDiagnosticsEveryItrAndSolutionAtTheEnd()
{
    m_IO->display(dotk::types::FINAL);
}

void DOTk_NonlinearCG::getMin()
{
    m_IO->license();
    m_IO->openFile("DOTk_NonLinearCGDiagnostics.out");

    this->initialize();

    size_t itr = 0;
    m_DataMng->setNumOptimizationItrDone(itr);
    m_IO->printDiagnosticsReport(m_LineSearch, m_DataMng);
    while(1)
    {
        m_DescentDirection->direction(m_DataMng);

        m_DataMng->storeCurrentState();
        m_LineSearch->solveSubProblem(m_DataMng);

        ++itr;
        m_DataMng->setNumOptimizationItrDone(itr);
        dotk::DOTk_FirstOrderAlgorithm::setNumItrDone(itr);
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

void DOTk_NonlinearCG::initialize()
{
    Real initial_objective_function_value = m_DataMng->evaluateObjective();
    m_DataMng->setNewObjectiveFunctionValue(initial_objective_function_value);

    m_DataMng->computeGradient();
    Real initial_norm_gradient = m_DataMng->getNewGradient()->norm();
    m_DataMng->setNormNewGradient(initial_norm_gradient);
}

}
