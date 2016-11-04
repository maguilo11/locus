/*
 * DOTk_TrustRegionInexactNewtonIO.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "vector.hpp"
#include "DOTk_TrustRegion.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_InexactNewtonAlgorithms.hpp"
#include "DOTk_TrustRegionInexactNewton.hpp"
#include "DOTk_TrustRegionInexactNewtonIO.hpp"
#include "DOTk_TrustRegionAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_TrustRegionInexactNewtonIO::DOTk_TrustRegionInexactNewtonIO() :
        m_PrintLicenseFlag(false),
        m_DiagnosticsFile(),
        m_DisplayFlag(dotk::types::display_t::OFF),
        m_TrustRegionSubProblemConverged(false)
{
}

DOTk_TrustRegionInexactNewtonIO::~DOTk_TrustRegionInexactNewtonIO()
{
}

void DOTk_TrustRegionInexactNewtonIO::display(dotk::types::display_t input_)
{
    m_DisplayFlag = input_;
}

dotk::types::display_t DOTk_TrustRegionInexactNewtonIO::display() const
{
    return (m_DisplayFlag);
}

void DOTk_TrustRegionInexactNewtonIO::license(bool input_)
{
    m_PrintLicenseFlag = input_;
}

void DOTk_TrustRegionInexactNewtonIO::license()
{
    if(m_PrintLicenseFlag == false)
    {
        return;
    }
    std::ostringstream msg;
    dotk::ioUtils::getLicenseMessage(msg);
    dotk::ioUtils::printMessage(msg);
}

void DOTk_TrustRegionInexactNewtonIO::openFile(const char * const name_)
{
    if(this->display() == dotk::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.open(name_, std::ios::out | std::ios::trunc);
}

void DOTk_TrustRegionInexactNewtonIO::closeFile()
{
    if(this->display() == dotk::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.close();
}

void DOTk_TrustRegionInexactNewtonIO::setTrustRegionSubProblemConvergedFlag(bool input_)
{
    m_TrustRegionSubProblemConverged = input_;
}

bool DOTk_TrustRegionInexactNewtonIO::didTrustRegionSubProblemConverged() const
{
    return (m_TrustRegionSubProblemConverged);
}

void DOTk_TrustRegionInexactNewtonIO::printDiagnosticReport(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
                                                            const std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                                                            const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & opt_mng_,
                                                            bool did_trust_region_subproblem_converged_)
{
    this->setTrustRegionSubProblemConvergedFlag(did_trust_region_subproblem_converged_);
    dotk::types::trustregion_t type = opt_mng_->getTrustRegion()->getTrustRegionType();
    if( (this->display() == dotk::types::OFF) || (type == dotk::types::TRUST_REGION_DISABLED) )
    {
        return;
    }

    switch(type)
    {
        case dotk::types::TRUST_REGION_DOGLEG:
        case dotk::types::TRUST_REGION_DOUBLE_DOGLEG:
        {
            this->writeDiagnostics(alg_, solver_, opt_mng_);
            break;
        }
        default:
        {
            std::ostringstream msg;
            msg << "DOTk WARNING: Invalid Trust-Region Step. Diagnostics were not printed.\n";
            dotk::ioUtils::printMessage(msg);
            break;
        }
    }

    if(this->display() == dotk::types::ITERATION)
    {
        dotk::printControl(opt_mng_->getNewPrimal());
    }
}

void DOTk_TrustRegionInexactNewtonIO::writeHeader()
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
            << std::setw(14) << std::right << "Krylov-Error"
            << std::setw(15) << std::right << "Krylov-Exit"
            << "\n" << std::flush;
}

void DOTk_TrustRegionInexactNewtonIO::writeDiagnostics(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_)
{
    size_t num_opt_itr_done = alg_->getNumItrDone();
    size_t num_trust_region_subproblem_itr = mng_->getTrustRegion()->getNumTrustRegionSubProblemItrDone();

    if(this->display() == dotk::types::OFF)
    {
        return;
    }
    else if((num_opt_itr_done == 0) && (num_trust_region_subproblem_itr == 0))
    {
        this->writeInitialDiagnostics(alg_, mng_);
    }
    else if((num_opt_itr_done > 0) && (num_trust_region_subproblem_itr == 1))
    {
        this->writeFullTrustRegionSubProblemDiagnostics(alg_, solver_, mng_);
    }
    else if(this->didTrustRegionSubProblemConverged() == false)
    {
        this->writeTrustRegionSubProblemDiagnostics(alg_, solver_, mng_);
    }
    else
    {
        this->writeFullTrustRegionSubProblemDiagnostics(alg_, solver_, mng_);
    }

    if(this->display() == dotk::types::ITERATION)
    {
        dotk::printControl(mng_->getNewPrimal());
    }
}

void DOTk_TrustRegionInexactNewtonIO::writeInitialDiagnostics
(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
 const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_)
{
    Real norm_grad = mng_->getNormNewGradient();
    Real objective_func_value = mng_->getNewObjectiveFunctionValue();

    size_t num_opt_itr_done = alg_->getNumItrDone();
    size_t objective_function_counter = mng_->getObjectiveFuncEvalCounter();

    this->writeHeader();
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

void DOTk_TrustRegionInexactNewtonIO::writeTrustRegionSubProblemDiagnostics
(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
 const std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
 const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_)
{
    Real trust_region_radius = mng_->getTrustRegionRadius();
    Real actual_reduction = mng_->getTrustRegion()->getActualReduction();
    Real predicted_reduction = mng_->getTrustRegion()->getPredictedReduction();
    Real ared_over_pred_ratio = actual_reduction / (predicted_reduction + std::numeric_limits<Real>::epsilon());

    size_t num_trust_region_subproblem_itr = mng_->getTrustRegion()->getNumTrustRegionSubProblemItrDone();

    std::ostringstream exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(solver_->getSolverStopCriterion(), exit_criterion);

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
            << std::setw(12) << std::right << solver_->getNumSolverItrDone()
            << std::setw(14) << std::right << solver_->getSolverResidualNorm()
            << std::setw(15) << std::right << exit_criterion.str().c_str()
            << "\n" << std::flush;
}

void DOTk_TrustRegionInexactNewtonIO::writeFullTrustRegionSubProblemDiagnostics
(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
 const std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
 const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_)
{
    Real norm_grad = mng_->getNormNewGradient();
    Real norm_trial_step = mng_->getNormTrialStep();
    Real objective_func_value = mng_->getNewObjectiveFunctionValue();

    Real trust_region_radius = mng_->getTrustRegionRadius();
    Real actual_reduction = mng_->getTrustRegion()->getActualReduction();
    Real predicted_reduction = mng_->getTrustRegion()->getPredictedReduction();
    Real ared_over_pred_ratio = actual_reduction / (predicted_reduction + std::numeric_limits<Real>::epsilon());

    size_t num_opt_itr_done = alg_->getNumItrDone();
    size_t objective_function_counter = mng_->getObjectiveFuncEvalCounter();
    size_t num_trust_region_subproblem_itr = mng_->getTrustRegion()->getNumTrustRegionSubProblemItrDone();

    std::ostringstream exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(solver_->getSolverStopCriterion(), exit_criterion);

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
            << std::setw(13) << std::right << ared_over_pred_ratio
            << std::setw(12) << std::right << solver_->getNumSolverItrDone()
            << std::setw(14) << std::right << solver_->getSolverResidualNorm()
            << std::setw(15) << std::right << exit_criterion.str().c_str()
            << "\n" << std::flush;
}

}
