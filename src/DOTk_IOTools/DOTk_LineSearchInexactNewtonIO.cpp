/*
 * DOTk_LineSearchInexactNewtonIO.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iomanip>
#include <fstream>
#include <sstream>

#include "vector.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_LineSearchStepMng.hpp"
#include "DOTk_LineSearchInexactNewton.hpp"
#include "DOTk_LineSearchInexactNewtonIO.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_LineSearchInexactNewtonIO::DOTk_LineSearchInexactNewtonIO() :
        m_PrintLicenseFlag(false),
        m_DiagnosticsFile(),
        m_DisplayFlag(dotk::types::display_t::OFF)
{
}

DOTk_LineSearchInexactNewtonIO::~DOTk_LineSearchInexactNewtonIO()
{
}

void DOTk_LineSearchInexactNewtonIO::display(dotk::types::display_t input_)
{
    m_DisplayFlag = input_;
}

dotk::types::display_t DOTk_LineSearchInexactNewtonIO::display() const
{
    return (m_DisplayFlag);
}

void DOTk_LineSearchInexactNewtonIO::license(bool input_)
{
    m_PrintLicenseFlag = input_;
}

void DOTk_LineSearchInexactNewtonIO::license()
{
    if(m_PrintLicenseFlag == false)
    {
        return;
    }
    std::ostringstream msg;
    dotk::ioUtils::getLicenseMessage(msg);
    dotk::ioUtils::printMessage(msg);
}

void DOTk_LineSearchInexactNewtonIO::openFile(const char * const name_)
{
    if(this->display() == dotk::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.open(name_, std::ios::out | std::ios::trunc);
}

void DOTk_LineSearchInexactNewtonIO::closeFile()
{
    if(this->display() == dotk::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.close();
}

void DOTk_LineSearchInexactNewtonIO::printDiagnosticsReport(const std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                                                            const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                                                            const std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_)
{
    if(this->display() == dotk::types::OFF)
    {
        return;
    }

    const Real norm_gradient = mng_->getNormNewGradient();
    const Real objective_func_value = mng_->getNewObjectiveFunctionValue();
    const Int objective_function_counter = mng_->getObjectiveFuncEvalCounter();
    const size_t num_optimization_itr_done = mng_->getNumOptimizationItrDone();
    if(num_optimization_itr_done == 0)
    {
        m_DiagnosticsFile << std::setw(10) << std::right << "Iteration" << std::setw(12) << std::right << "Func-count"
                << std::setw(12) << std::right << "F(x)" << std::setw(12) << std::right << "norm(G)" << std::setw(12)
                << std::right << "norm(P)" << std::setw(12) << std::right << "Krylov-Itr" << std::setw(14) << std::right
                << "Krylov-Error" << std::setw(15) << std::right << "Krylov-Exit" << std::setw(15) << std::right
                << "LineSrch-Step" << std::setw(14) << std::right << "LineSrch-Itr" << "\n" << std::flush;

        m_DiagnosticsFile << std::setw(10) << std::right << std::scientific << std::setprecision(4)
                << num_optimization_itr_done << std::setw(12) << std::right << objective_function_counter
                << std::setw(12) << std::right << objective_func_value << std::setw(12) << std::right << norm_gradient
                << std::setw(12) << std::right << "*" << std::setw(12) << std::right << "*" << std::setw(14)
                << std::right << "*" << std::setw(15) << std::right << "*" << std::setw(15) << std::right << "*"
                << std::setw(14) << std::right << "*" << "\n" << std::flush;
    }
    else
    {
        const Real norm_trial_step = mng_->getNormTrialStep();
        std::ostringstream exit_criterion;
        dotk::ioUtils::getSolverExitCriterion(solver_->getSolverStopCriterion(), exit_criterion);

        m_DiagnosticsFile << std::setw(10) << std::right << std::scientific << std::setprecision(4)
                << num_optimization_itr_done << std::setw(12) << std::right << objective_function_counter
                << std::setw(12) << std::right << objective_func_value << std::setw(12) << std::right << norm_gradient
                << std::setw(12) << std::right << norm_trial_step << std::setw(12) << std::right
                << solver_->getNumSolverItrDone() << std::setw(14) << std::right << solver_->getSolverResidualNorm()
                << std::setw(15) << std::right << exit_criterion.str().c_str() << std::setw(15) << std::right
                << step_->step() << std::setw(14) << std::right << step_->iterations() << "\n" << std::flush;
    }

    if(this->display() == dotk::types::ITERATION)
    {
        dotk::printControl(mng_->getNewPrimal());
    }
}

}
