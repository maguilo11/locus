/*
 * TRROM_TrustRegionNewtonIO.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iomanip>
#include <sstream>

#include "TRROM_Vector.hpp"
#include "TRROM_VariablesUtils.hpp"
#include "TRROM_TrustRegionStepMng.hpp"
#include "TRROM_SteihaugTointSolver.hpp"
#include "TRROM_OptimizationDataMng.hpp"
#include "TRROM_TrustRegionNewtonIO.hpp"

namespace trrom
{

TrustRegionNewtonIO::TrustRegionNewtonIO() :
        m_NumOptimizationItrDone(0),
        m_DiagnosticsFile(),
        m_DisplayType(trrom::types::OFF)

{
}

TrustRegionNewtonIO::~TrustRegionNewtonIO()
{
}

void TrustRegionNewtonIO::setNumOptimizationItrDone(int itr_)
{
    m_NumOptimizationItrDone = itr_;
}

int TrustRegionNewtonIO::getNumOptimizationItrDone() const
{
    return (m_NumOptimizationItrDone);
}

void TrustRegionNewtonIO::printOutputPerIteration()
{
    m_DisplayType = trrom::types::ITERATION;
}

void TrustRegionNewtonIO::printOutputFinalIteration()
{
    m_DisplayType = trrom::types::FINAL;
}

void TrustRegionNewtonIO::setDisplayOption(trrom::types::display_t option_)
{
    m_DisplayType = option_;
}

trrom::types::display_t TrustRegionNewtonIO::getDisplayOption() const
{
    return (m_DisplayType);
}

void TrustRegionNewtonIO::openFile(const std::string & name_)
{
    if(this->getDisplayOption() == trrom::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.open(name_.c_str(), std::ios::out | std::ios::trunc);
}

void TrustRegionNewtonIO::closeFile()
{
    if(this->getDisplayOption() == trrom::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.close();
}

void TrustRegionNewtonIO::printInitialDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_)
{
    if(this->getDisplayOption() == trrom::types::OFF)
    {
        return;
    }

    double norm_grad = data_mng_->getNormNewGradient();
    double objective_func_value = data_mng_->getNewObjectiveFunctionValue();

    int num_opt_itr_done = this->getNumOptimizationItrDone();
    int objective_function_counter = data_mng_->getObjectiveFunctionEvaluationCounter();

    this->printHeader();
    m_DiagnosticsFile << std::setw(10) << std::right << std::scientific << std::setprecision(4) << num_opt_itr_done
            << std::setw(12) << std::right << objective_function_counter << std::setw(12) << std::right
            << objective_func_value << std::setw(12) << std::right << norm_grad << std::setw(12) << std::right << "*"
            << std::setw(8) << std::right << "*" << std::setw(12) << std::right << "*" << std::setw(13) << std::right
            << "*" << std::setw(13) << std::right << "*" << std::setw(13) << std::right << "*" << std::setw(12)
            << std::right << "*" << std::setw(14) << std::right << "*" << std::setw(15) << std::right << "*" << "\n"
            << std::flush;
}

void TrustRegionNewtonIO::printSolution(const std::shared_ptr<trrom::Vector<double> > & primal_)
{
    if(this->getDisplayOption() == trrom::types::OFF)
    {
        return;
    }
    trrom::printSolution(primal_);
}

void TrustRegionNewtonIO::printConvergedDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                                      const std::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                                      const trrom::TrustRegionStepMng * const step_mng_)
{
    this->printSubProblemFirstItrDiagnostics(data_mng_, solver_, step_mng_);
}

void TrustRegionNewtonIO::printTrustRegionSubProblemDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                                                  const std::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                                                  const trrom::TrustRegionStepMng * const step_mng_)
{
    int num_opt_itr_done = this->getNumOptimizationItrDone();
    int num_subproblem_itr = step_mng_->getNumTrustRegionSubProblemItrDone();

    if(this->getDisplayOption() == trrom::types::OFF)
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

void TrustRegionNewtonIO::printHeader()
{
    m_DiagnosticsFile << std::setw(10) << std::right << "Iteration" << std::setw(12) << std::right << "Func-count"
            << std::setw(12) << std::right << "F(x)" << std::setw(12) << std::right << "norm(G)" << std::setw(12)
            << std::right << "norm(P)" << std::setw(8) << std::right << "TR-Itr" << std::setw(12) << std::right
            << "TR-Radius" << std::setw(13) << std::right << "ActualReduc" << std::setw(13) << std::right << "PredReduc"
            << std::setw(13) << std::right << "Ratio" << std::setw(12) << std::right << "Krylov-Itr" << std::setw(14)
            << std::right << "Krylov-Res" << std::setw(15) << std::right << "Krylov-Exit" << "\n" << std::flush;
}
void TrustRegionNewtonIO::printSubProblemDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                                       const std::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                                       const trrom::TrustRegionStepMng * const step_mng_)
{
    double actual_reduction = step_mng_->getActualReduction();
    double predicted_reduction = step_mng_->getPredictedReduction();
    double trust_region_radius = step_mng_->getTrustRegionRadius();
    double ared_over_pred_ratio = step_mng_->getActualOverPredictedReduction();
    int num_trust_region_subproblem_itr = step_mng_->getNumTrustRegionSubProblemItrDone();

    std::ostringstream exit_criterion;
    this->getSolverExitCriterion(solver_->getStoppingCriterion(), exit_criterion);

    m_DiagnosticsFile << std::setw(10) << std::right << std::scientific << std::setprecision(4) << "*" << std::setw(12)
            << std::right << "*" << std::setw(12) << std::right << "*" << std::setw(12) << std::right << "*"
            << std::setw(12) << std::right << "*" << std::setw(8) << std::right << num_trust_region_subproblem_itr
            << std::setw(12) << std::right << trust_region_radius << std::setw(13) << std::right << actual_reduction
            << std::setw(13) << std::right << predicted_reduction << std::setw(13) << std::right << ared_over_pred_ratio
            << std::setw(12) << std::right << solver_->getNumItrDone() << std::setw(14) << std::right
            << solver_->getResidualNorm() << std::setw(15) << std::right << exit_criterion.str().c_str() << "\n"
            << std::flush;
}

void TrustRegionNewtonIO::printSubProblemFirstItrDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                                               const std::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                                               const trrom::TrustRegionStepMng * const step_mng_)
{
    double norm_grad = data_mng_->getNormNewGradient();
    double norm_trial_step = data_mng_->getNormTrialStep();
    double objective_func_value = data_mng_->getNewObjectiveFunctionValue();

    double trust_region_radius = step_mng_->getTrustRegionRadius();
    double actual_reduction = step_mng_->getActualReduction();
    double predicted_reduction = step_mng_->getPredictedReduction();
    double areduc_over_preduc_ratio = step_mng_->getActualOverPredictedReduction();

    int num_opt_itr_done = this->getNumOptimizationItrDone();
    int objective_function_counter = data_mng_->getObjectiveFunctionEvaluationCounter();
    int num_trust_region_subproblem_itr = step_mng_->getNumTrustRegionSubProblemItrDone();

    std::ostringstream solver_stopping_criterion;
    this->getSolverExitCriterion(solver_->getStoppingCriterion(), solver_stopping_criterion);

    m_DiagnosticsFile << std::setw(10) << std::right << std::scientific << std::setprecision(4) << num_opt_itr_done
            << std::setw(12) << std::right << objective_function_counter << std::setw(12) << std::right
            << objective_func_value << std::setw(12) << std::right << norm_grad << std::setw(12) << std::right
            << norm_trial_step << std::setw(8) << std::right << num_trust_region_subproblem_itr << std::setw(12)
            << std::right << trust_region_radius << std::setw(13) << std::right << actual_reduction << std::setw(13)
            << std::right << predicted_reduction << std::setw(13) << std::right << areduc_over_preduc_ratio
            << std::setw(12) << std::right << solver_->getNumItrDone() << std::setw(14) << std::right
            << solver_->getResidualNorm() << std::setw(15) << std::right << solver_stopping_criterion.str().c_str()
            << "\n" << std::flush;
}

void TrustRegionNewtonIO::printCurrentSolution(const std::shared_ptr<trrom::Vector<double> > & primal_)
{
    if(this->getDisplayOption() == trrom::types::OFF)
    {
        return;
    }
    else if(this->getDisplayOption() == trrom::types::ITERATION)
    {
        trrom::printSolution(primal_);
    }
}

void TrustRegionNewtonIO::getSolverExitCriterion(trrom::types::solver_stop_criterion_t type_,
                                                   std::ostringstream & criterion_)
{
    switch(type_)
    {
        case trrom::types::NaN_CURVATURE_DETECTED:
        {
            criterion_ << "NaNCurv";
            break;
        }
        case trrom::types::ZERO_CURVATURE_DETECTED:
        {
            criterion_ << "ZeroCurv";
            break;
        }
        case trrom::types::NEGATIVE_CURVATURE_DETECTED:
        {
            criterion_ << "NegCurv";
            break;
        }
        case trrom::types::INF_CURVATURE_DETECTED:
        {
            criterion_ << "InfCurv";
            break;
        }
        case trrom::types::SOLVER_TOLERANCE_SATISFIED:
        {
            criterion_ << "Tolerance";
            break;
        }
        case trrom::types::TRUST_REGION_VIOLATED:
        {
            criterion_ << "TrustReg";
            break;
        }
        case trrom::types::MAX_SOLVER_ITR_REACHED:
        {
            criterion_ << "MaxItr";
            break;
        }
        case trrom::types::SOLVER_DID_NOT_CONVERGED:
        {
            criterion_ << "NotCnvg";
            break;
        }
        case trrom::types::NaN_RESIDUAL_NORM:
        {
            criterion_ << "NaNResNorm";
            break;
        }
        case trrom::types::INF_RESIDUAL_NORM:
        {
            criterion_ << "InfResNorm";
            break;
        }
        case trrom::types::INVALID_INEXACTNESS_MEASURE:
        {
            criterion_ << "InvalInxMeas";
            break;
        }
        case trrom::types::INVALID_ORTHOGONALITY_MEASURE:
        {
            criterion_ << "InvlOrthoMeas";
            break;
        }
    }
}

}
