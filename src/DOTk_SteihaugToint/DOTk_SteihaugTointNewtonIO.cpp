/*
 * DOTk_SteihaugTointNewtonIO.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iomanip>
#include <sstream>

#include "vector.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_SteihaugTointSolver.hpp"
#include "DOTk_TrustRegionStepMng.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_SteihaugTointNewtonIO.hpp"

namespace dotk
{

DOTk_SteihaugTointNewtonIO::DOTk_SteihaugTointNewtonIO() :
        m_NumOptimizationItrDone(0),
        m_DiagnosticsFile(),
        m_DisplayType(dotk::types::OFF)

{
}

DOTk_SteihaugTointNewtonIO::~DOTk_SteihaugTointNewtonIO()
{
}

void DOTk_SteihaugTointNewtonIO::setNumOptimizationItrDone(size_t itr_)
{
    m_NumOptimizationItrDone = itr_;
}

size_t DOTk_SteihaugTointNewtonIO::getNumOptimizationItrDone() const
{
    return (m_NumOptimizationItrDone);
}

void DOTk_SteihaugTointNewtonIO::printOutputPerIteration()
{
    m_DisplayType = dotk::types::ITERATION;
}

void DOTk_SteihaugTointNewtonIO::printOutputFinalIteration()
{
    m_DisplayType = dotk::types::FINAL;
}

void DOTk_SteihaugTointNewtonIO::setDisplayOption(dotk::types::display_t option_)
{
    m_DisplayType = option_;
}

dotk::types::display_t DOTk_SteihaugTointNewtonIO::getDisplayOption() const
{
    return (m_DisplayType);
}

void DOTk_SteihaugTointNewtonIO::openFile(const char * const name_)
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.open(name_, std::ios::out | std::ios::trunc);
}

void DOTk_SteihaugTointNewtonIO::closeFile()
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.close();
}

void DOTk_SteihaugTointNewtonIO::printInitialDiagnostics(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_)
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }

    Real norm_grad = data_mng_->getNormNewGradient();
    Real objective_func_value = data_mng_->getNewObjectiveFunctionValue();

    size_t num_opt_itr_done = this->getNumOptimizationItrDone();
    size_t objective_function_counter = data_mng_->getObjectiveFunctionEvaluationCounter();

    this->printHeader();
    m_DiagnosticsFile << std::setw(10) << std::right
            << std::scientific << std::setprecision(4) << num_opt_itr_done
            << std::setw(12) << std::right << objective_function_counter
            << std::setw(12) << std::right << objective_func_value
            << std::setw(12) << std::right << norm_grad
            << std::setw(12) << std::right << "*"
            << std::setw(8) << std::right << "*"
            << std::setw(12) << std::right << "*"
            << std::setw(13) << std::right << "*"
            << std::setw(13) << std::right << "*"
            << std::setw(13) << std::right << "*"
            << std::setw(12) << std::right << "*"
            << std::setw(14) << std::right << "*"
            << std::setw(15) << std::right << "*"
            << "\n" << std::flush;
}

void DOTk_SteihaugTointNewtonIO::printSolution(const std::shared_ptr<dotk::Vector<Real> > & primal_)
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }
    dotk::printSolution(primal_);
}

void DOTk_SteihaugTointNewtonIO::printConvergedDiagnostics(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                                           const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                                           const dotk::DOTk_TrustRegionStepMng * const step_mng_)
{
    this->printSubProblemFirstItrDiagnostics(data_mng_, solver_, step_mng_);
}

void DOTk_SteihaugTointNewtonIO::printTrustRegionSubProblemDiagnostics(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                                                       const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                                                       const dotk::DOTk_TrustRegionStepMng * const step_mng_)
{
    size_t num_opt_itr_done = this->getNumOptimizationItrDone();
    size_t num_subproblem_itr = step_mng_->getNumTrustRegionSubProblemItrDone();

    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }
    else if((num_opt_itr_done >= 0) && (num_subproblem_itr == 1))
    {
        this->printCurrentSolution(data_mng_->getNewPrimal());
        this->printSubProblemFirstItrDiagnostics(data_mng_, solver_, step_mng_);
    }
    else
    {
        this->printSubProblemDiagnostics(data_mng_, solver_, step_mng_);
    }
}

void DOTk_SteihaugTointNewtonIO::printHeader()
{
    m_DiagnosticsFile << std::setw(10) << std::right << "Iteration"
            << std::setw(12) << std::right << "Func-count"
            << std::setw(12) << std::right << "F(x)"
            << std::setw(12) << std::right << "norm(G)"
            << std::setw(12) << std::right << "norm(P)"
            << std::setw(8) << std::right << "TR-Itr"
            << std::setw(12) << std::right << "TR-Radius"
            << std::setw(13) << std::right << "ActualReduc"
            << std::setw(13) << std::right << "PredReduc"
            << std::setw(13) << std::right << "Ratio"
            << std::setw(12) << std::right << "Krylov-Itr"
            << std::setw(14) << std::right << "Krylov-Res"
            << std::setw(15) << std::right << "Krylov-Exit"
            << "\n" << std::flush;
}
void DOTk_SteihaugTointNewtonIO::printSubProblemDiagnostics(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                                            const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                                            const dotk::DOTk_TrustRegionStepMng * const step_mng_)
{
    Real actual_reduction = step_mng_->getActualReduction();
    Real predicted_reduction = step_mng_->getPredictedReduction();
    Real trust_region_radius = step_mng_->getTrustRegionRadius();
    Real ared_over_pred_ratio = step_mng_->getActualOverPredictedReduction();
    size_t num_trust_region_subproblem_itr = step_mng_->getNumTrustRegionSubProblemItrDone();

    std::ostringstream exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(solver_->getStoppingCriterion(), exit_criterion);

    m_DiagnosticsFile << std::setw(10) << std::right
            << std::scientific << std::setprecision(4) << "*"
            << std::setw(12) << std::right << "*"
            << std::setw(12) << std::right << "*"
            << std::setw(12) << std::right << "*"
            << std::setw(12) << std::right << "*"
            << std::setw(8) << std::right << num_trust_region_subproblem_itr
            << std::setw(12) << std::right << trust_region_radius
            << std::setw(13) << std::right << actual_reduction
            << std::setw(13) << std::right << predicted_reduction
            << std::setw(13) << std::right << ared_over_pred_ratio
            << std::setw(12) << std::right << solver_->getNumItrDone()
            << std::setw(14) << std::right << solver_->getResidualNorm()
            << std::setw(15) << std::right << exit_criterion.str().c_str()
            << "\n" << std::flush;
}

void DOTk_SteihaugTointNewtonIO::printSubProblemFirstItrDiagnostics(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                                                    const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                                                    const dotk::DOTk_TrustRegionStepMng * const step_mng_)
{
    Real norm_grad = data_mng_->getNormNewGradient();
    Real norm_trial_step = data_mng_->getNormTrialStep();
    Real objective_func_value = data_mng_->getNewObjectiveFunctionValue();

    Real trust_region_radius = step_mng_->getTrustRegionRadius();
    Real actual_reduction = step_mng_->getActualReduction();
    Real predicted_reduction = step_mng_->getPredictedReduction();
    Real areduc_over_preduc_ratio = step_mng_->getActualOverPredictedReduction();

    size_t num_opt_itr_done = this->getNumOptimizationItrDone();
    size_t objective_function_counter = data_mng_->getObjectiveFunctionEvaluationCounter();
    size_t num_trust_region_subproblem_itr = step_mng_->getNumTrustRegionSubProblemItrDone();

    std::ostringstream solver_stopping_criterion;
    dotk::ioUtils::getSolverExitCriterion(solver_->getStoppingCriterion(), solver_stopping_criterion);

    m_DiagnosticsFile << std::setw(10) << std::right
            << std::scientific << std::setprecision(4) << num_opt_itr_done
            << std::setw(12) << std::right << objective_function_counter
            << std::setw(12) << std::right << objective_func_value
            << std::setw(12) << std::right << norm_grad
            << std::setw(12) << std::right << norm_trial_step
            << std::setw(8) << std::right << num_trust_region_subproblem_itr
            << std::setw(12) << std::right << trust_region_radius
            << std::setw(13) << std::right << actual_reduction
            << std::setw(13) << std::right << predicted_reduction
            << std::setw(13) << std::right << areduc_over_preduc_ratio
            << std::setw(12) << std::right << solver_->getNumItrDone()
            << std::setw(14) << std::right << solver_->getResidualNorm()
            << std::setw(15) << std::right << solver_stopping_criterion.str().c_str()
            << "\n" << std::flush;
}

void DOTk_SteihaugTointNewtonIO::printCurrentSolution
(const std::shared_ptr<dotk::Vector<Real> > & primal_)
{
    if(this->getDisplayOption() == dotk::types::OFF)
    {
        return;
    }
    else if(this->getDisplayOption() == dotk::types::ITERATION)
    {
        dotk::printSolution(primal_);
    }
}

}
