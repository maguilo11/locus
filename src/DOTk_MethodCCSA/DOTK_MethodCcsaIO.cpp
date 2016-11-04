/*
 * DOTK_MethodCcsaIO.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iomanip>
#include <sstream>

#include "DOTk_DataMngCCSA.hpp"
#include "DOTK_MethodCcsaIO.hpp"
#include "DOTk_AlgorithmCCSA.hpp"
#include "DOTk_VariablesUtils.hpp"

namespace dotk
{

DOTK_MethodCcsaIO::DOTK_MethodCcsaIO() :
        m_OutputFileStream(),
        m_DisplayType(dotk::types::OFF)
{
}

DOTK_MethodCcsaIO::~DOTK_MethodCcsaIO()
{
}

dotk::types::display_t DOTK_MethodCcsaIO::getDisplayOption() const
{
    return (m_DisplayType);
}

void DOTK_MethodCcsaIO::printSolutionAtEachIteration()
{
    m_DisplayType = dotk::types::ITERATION;
}

void DOTK_MethodCcsaIO::printSolutionAtFinalIteration()
{
    m_DisplayType = dotk::types::FINAL;
}

void DOTK_MethodCcsaIO::openFile(const char * const name_)
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }
    m_OutputFileStream.open(name_, std::ios::out | std::ios::trunc);
    m_OutputFileStream.precision(5);
}

void DOTK_MethodCcsaIO::closeFile()
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }
    m_OutputFileStream.close();
}

void DOTK_MethodCcsaIO::printHeader()
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }

    m_OutputFileStream << std::setw(10) << std::right << "Iteration"
            << std::setw(15) << "Func-count"
            << std::setw(14) << "F(x)"
            << std::setw(17) << "norm(G)"
            << std::setw(18) << "DeltaCntrl"
            << std::setw(15) << "max(Res)"
            << std::setw(20) << "max(FeasMeas)"
            << "\n" << std::flush;
    m_OutputFileStream << "-----------"
            << std::setw(15) << "------------"
            << std::setw(16) << "-----------"
            << std::setw(16) << "-----------"
            << std::setw(17) << "------------"
            << std::setw(16) << "-----------"
            << std::setw(18) << "-------------"
            << "\n" << std::flush;
}

void DOTK_MethodCcsaIO::printSolution(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }

    dotk::printDual(data_mng_->m_Dual);
    dotk::printControl(data_mng_->m_CurrentControl);
}

void DOTK_MethodCcsaIO::print(const dotk::DOTk_AlgorithmCCSA* const algorithm_,
           const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }

    if(algorithm_->getIterationCount() < 1)
    {
        this->printHeader();
    }

    m_OutputFileStream << std::setw(6) << std::right
            << std::scientific << algorithm_->getIterationCount()
            << std::setw(14) << data_mng_->getObjectiveFunctionEvaluationCounter()
            << std::setw(22) << data_mng_->m_CurrentObjectiveFunctionValue
            << std::setw(16) << algorithm_->getCurrentObjectiveGradientNorm()
            << std::setw(17) << algorithm_->getCurrentControlStagnationMeasure()
            << std::setw(16) << algorithm_->getCurrentMaxResidual()
             << std::setw(17) << algorithm_->getCurrentMaxFeasibilityMeasure()
            << "\n" << std::flush;

    if(this->getDisplayOption() == dotk::types::ITERATION)
    {
        dotk::printDual(data_mng_->m_Dual);
        dotk::printControl(data_mng_->m_CurrentControl);
    }
}

}
