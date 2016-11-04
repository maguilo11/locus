/*
 * DOTk_FirstOrderLineSearchAlgIO.cpp
 *
 *  Created on: Sep 18, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <fstream>
#include <iomanip>
#include <sstream>

#include "vector.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_LineSearchStepMng.hpp"
#include "DOTk_FirstOrderLineSearchAlgIO.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

using namespace dotk;

DOTk_FirstOrderLineSearchAlgIO::DOTk_FirstOrderLineSearchAlgIO() :
        m_PrintLicenseFlag(false),
        m_DiagnosticsFile(),
        m_DisplayFlag(dotk::types::display_t::OFF)
{
}

DOTk_FirstOrderLineSearchAlgIO::~DOTk_FirstOrderLineSearchAlgIO()
{
}

void DOTk_FirstOrderLineSearchAlgIO::display(dotk::types::display_t input_)
{
    m_DisplayFlag = input_;
}

dotk::types::display_t DOTk_FirstOrderLineSearchAlgIO::display() const
{
    return (m_DisplayFlag);
}

void DOTk_FirstOrderLineSearchAlgIO::license(bool input_)
{
    m_PrintLicenseFlag = input_;
}

void DOTk_FirstOrderLineSearchAlgIO::license()
{
    if(m_PrintLicenseFlag == false)
    {
        return;
    }
    std::ostringstream msg;
    dotk::ioUtils::getLicenseMessage(msg);
    dotk::ioUtils::printMessage(msg);
}

void DOTk_FirstOrderLineSearchAlgIO::openFile(const char* const name_)
{
    if(this->display() == dotk::types::display_t::OFF)
    {
        return;
    }
    m_DiagnosticsFile.open(name_, std::ios::out | std::ios::trunc);
}

void DOTk_FirstOrderLineSearchAlgIO::closeFile()
{
    if(this->display() == dotk::types::display_t::OFF)
    {
        return;
    }
    m_DiagnosticsFile.close();
}

void DOTk_FirstOrderLineSearchAlgIO::printDiagnosticsReport(const std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                                                            const std::tr1::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_)
{
    if(this->display() == dotk::types::display_t::OFF)
    {
        return;
    }

    const Real norm_gradient = mng_->getNormNewGradient();
    const Real objective_func_value = mng_->getNewObjectiveFunctionValue();
    const size_t num_optimization_itr_done = mng_->getNumOptimizationItrDone();
    if(num_optimization_itr_done == 0)
    {
        m_DiagnosticsFile << std::setw(10) << std::right << "Iteration"
                << std::setw(12) << std::right << "Func-count"
                << std::setw(12) << std::right << "F(x)"
                << std::setw(12) << std::right << "norm(G)"
                << std::setw(12) << std::right << "norm(P)"
                << std::setw(15) << std::right << "LineSrch-Step"
                << std::setw(14) << std::right << "LineSrch-Itr" << "\n" << std::flush;
        m_DiagnosticsFile << std::setw(10) << std::right << std::scientific << std::setprecision(4) << num_optimization_itr_done
        << std::setw(12) << std::right << mng_->getRoutinesMng()->getObjectiveFunctionEvaluationCounter()
        << std::setw(12) << std::right << objective_func_value
        << std::setw(12) << std::right << norm_gradient
        << std::setw(12) << std::right << "*"
        << std::setw(15) << std::right << "*"
        << std::setw(14) << std::right << "*" << "\n" << std::flush;
    }
    else
    {
        const Real trial_step_norm = mng_->getNormTrialStep();
        m_DiagnosticsFile << std::setw(10) << std::right << std::scientific << std::setprecision(4) << num_optimization_itr_done
        << std::setw(12) << std::right << mng_->getRoutinesMng()->getObjectiveFunctionEvaluationCounter()
        << std::setw(12) << std::right << objective_func_value
        << std::setw(12) << std::right << norm_gradient
        << std::setw(12) << std::right << trial_step_norm
        << std::setw(15) << std::right << step_->step()
        << std::setw(14) << std::right << step_->iterations() << "\n" << std::flush;
    }
    bool print_solution = this->display() == dotk::types::display_t::ITERATION ? true : false;
    if(print_solution == true)
    {
        dotk::printControl(mng_->getNewPrimal());
    }
}
