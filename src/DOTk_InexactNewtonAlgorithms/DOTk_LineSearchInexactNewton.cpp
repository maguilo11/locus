/*
 * DOTk_LineSearchInexactNewton.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <fstream>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_LineSearchStepMng.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_KrylovSolverFactory.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_LineSearchInexactNewton.hpp"
#include "DOTk_LineSearchInexactNewtonIO.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_LineSearchInexactNewton::DOTk_LineSearchInexactNewton(const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                           const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & aStep,
                                                           const std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & aMng) :
        dotk::DOTk_InexactNewtonAlgorithms(dotk::types::LINE_SEARCH_INEXACT_NEWTON),
        m_SolverRhsVector(aMng->getTrialStep()->clone()),
        m_KrylovSolver(),
        m_IO(std::make_shared<dotk::DOTk_LineSearchInexactNewtonIO>()),
        m_LineSearch(aStep),
        m_LinearOperator(aLinearOperator),
        m_DataMng(aMng)
{
}

DOTk_LineSearchInexactNewton::~DOTk_LineSearchInexactNewton()
{
}

void DOTk_LineSearchInexactNewton::setNumItrDone(size_t aInput)
{
    dotk::DOTk_InexactNewtonAlgorithms::setNumItrDone(aInput);
    m_LinearOperator->setNumOtimizationItrDone(aInput);
    m_KrylovSolver->getDataMng()->getLeftPrec()->setNumOptimizationItrDone(aInput);
}

void DOTk_LineSearchInexactNewton::setMaxNumKrylovSolverItr(size_t aInput)
{
    m_KrylovSolver->setMaxNumKrylovSolverItr(aInput);
}

void DOTk_LineSearchInexactNewton::setPrecGmresKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                            size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildPrecGmresSolver(aPrimal, m_LinearOperator, aMaxNumIterations, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(aMaxNumIterations);
}

void DOTk_LineSearchInexactNewton::setLeftPrecCgKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                             size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgSolver(aPrimal, m_LinearOperator, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(aMaxNumIterations);
}

void DOTk_LineSearchInexactNewton::setLeftPrecCrKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                             size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCrSolver(aPrimal, m_LinearOperator, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(aMaxNumIterations);
}

void DOTk_LineSearchInexactNewton::setLeftPrecGcrKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                              size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecGcrSolver(aPrimal, m_LinearOperator, aMaxNumIterations, m_KrylovSolver);
}

void DOTk_LineSearchInexactNewton::setLeftPrecCgneKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                               size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgneSolver(aPrimal, m_LinearOperator, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(aMaxNumIterations);
}

void DOTk_LineSearchInexactNewton::setLeftPrecCgnrKrylovSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                               size_t aMaxNumIterations)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgnrSolver(aPrimal, m_LinearOperator, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(aMaxNumIterations);
}

void DOTk_LineSearchInexactNewton::printDiagnosticsAndSolutionEveryItr()
{
    m_IO->display(dotk::types::ITERATION);
}

void DOTk_LineSearchInexactNewton::printDiagnosticsEveryItrAndSolutionAtTheEnd()
{
    m_IO->display(dotk::types::FINAL);
}

void DOTk_LineSearchInexactNewton::getMin()
{
    m_IO->license();
    m_IO->openFile("DOTk_LineSearchNewtonCGDiagnostics.out");
    this->checkAlgorithmInputs();

    Real new_objective_func_value = m_DataMng->evaluateObjective();
    m_DataMng->setNewObjectiveFunctionValue(new_objective_func_value);
    m_DataMng->computeGradient();
    Real initial_norm_gradient = m_DataMng->getNewGradient()->norm();
    m_DataMng->setNormNewGradient(initial_norm_gradient);

    size_t itr = 0;
    m_IO->printDiagnosticsReport(m_KrylovSolver, m_LineSearch, m_DataMng);
    while(1)
    {
        ++itr;
        this->setNumItrDone(itr);

        dotk::gtools::getSteepestDescent(m_DataMng->getNewGradient(), m_SolverRhsVector);
        m_KrylovSolver->solve(m_SolverRhsVector, m_Criterion, m_DataMng);

        dotk::DOTk_InexactNewtonAlgorithms::setTrialStep(m_KrylovSolver, m_DataMng);
        dotk::gtools::checkDescentDirection(m_DataMng->getNewGradient(),
                                            m_DataMng->getTrialStep(),
                                            dotk::DOTk_InexactNewtonAlgorithms::getMinCosineAngleTol());

        m_DataMng->storeCurrentState();
        m_LineSearch->solveSubProblem(m_DataMng);

        m_LinearOperator->updateLimitedMemoryStorage(true);
        m_IO->printDiagnosticsReport(m_KrylovSolver, m_LineSearch, m_DataMng);

        if(dotk::DOTk_InexactNewtonAlgorithms::checkStoppingCriteria(m_DataMng) == true)
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

void DOTk_LineSearchInexactNewton::checkAlgorithmInputs()
{
    if(m_KrylovSolver.use_count() == 0)
    {
        std::perror("\n**** Error in DOTk_LineSearchInexactNewton::checkAlgorithmInputs -> User did not define Krylov solver. ABORT. ****\n");
        std::abort();
    }
}

}
