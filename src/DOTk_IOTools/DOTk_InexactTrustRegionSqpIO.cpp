/*
 * DOTk_InexactTrustRegionSqpIO.cpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iomanip>
#include <fstream>
#include <sstream>

#include "vector.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_InexactTrustRegionSQP.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"
#include "DOTk_InexactTrustRegionSqpIO.hpp"
#include "DOTk_TrustRegionAlgorithmsDataMng.hpp"
#include "DOTk_SequentialQuadraticProgramming.hpp"
#include "DOTk_InexactTrustRegionSqpSolverMng.hpp"

namespace dotk
{

DOTk_InexactTrustRegionSqpIO::DOTk_InexactTrustRegionSqpIO() :
        m_PrintLicenseFlag(false),
        m_DiagnosticsFile(),
        m_DisplayFlag(dotk::types::display_t::OFF)
{
}

DOTk_InexactTrustRegionSqpIO::~DOTk_InexactTrustRegionSqpIO()
{
}

void DOTk_InexactTrustRegionSqpIO::display(dotk::types::display_t input_)
{
    m_DisplayFlag = input_;
}

dotk::types::display_t DOTk_InexactTrustRegionSqpIO::display() const
{
    return (m_DisplayFlag);
}

void DOTk_InexactTrustRegionSqpIO::license(bool input_)
{
    m_PrintLicenseFlag = input_;
}

void DOTk_InexactTrustRegionSqpIO::license()
{
    if(m_PrintLicenseFlag == false)
    {
        return;
    }
    std::ostringstream msg;
    dotk::ioUtils::getLicenseMessage(msg);
    dotk::ioUtils::printMessage(msg);
}

void DOTk_InexactTrustRegionSqpIO::openFile(const char * const name_)
{
    if(this->display() == dotk::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.open(name_, std::ios::out | std::ios::trunc);
}

void DOTk_InexactTrustRegionSqpIO::closeFile()
{
    if(this->display() == dotk::types::OFF)
    {
        return;
    }
    m_DiagnosticsFile.close();
}

void DOTk_InexactTrustRegionSqpIO::printDiagnosticsReport(const dotk::DOTk_InexactTrustRegionSQP* const alg_,
                                                          const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_,
                                                          const std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng> & solver_)
{
    size_t num_opt_itr_done = alg_->getNumItrDone();
    size_t num_trust_region_subproblem_itr = alg_->getNumTrustRegionSubProblemItrDone();

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
        this->writeFirstTrustRegionSubProblemItrDiagnostics(alg_, mng_, solver_);
    }
    else
    {
        this->writeTrustRegionSubProblemDiagnostics(alg_, mng_, solver_);
    }

    if(this->display() == dotk::types::ITERATION)
    {
        dotk::printSolution(mng_->getNewPrimal());
    }
}

void DOTk_InexactTrustRegionSqpIO::writeHeader()
{
    m_DiagnosticsFile << std::setw(6) << std::right << "OptItr" << std::setw(12) << std::right << "Func-count"
            << std::setw(12) << std::right << "F(x)" << std::setw(12) << std::right << "norm(G)" << std::setw(12)
            << std::right << "norm(C)" << std::setw(12) << std::right << "norm(P)" << std::setw(9) << std::right
            << "TR_Itr" << std::setw(12) << std::right << "TR_Radius" << std::setw(13) << std::right << "Actual"
            << std::setw(13) << std::right << "Pred" << std::setw(12) << std::right << "Ratio" << std::setw(10)
            << std::right << "TangItr" << std::setw(13) << std::right << "norm(Res)" << std::setw(15) << std::right
            << "TangExit" << std::setw(14) << std::right << "DualItr" << std::setw(15) << std::right << "norm(Res)"
            << std::setw(12) << std::right << "DualExit" << std::setw(14) << std::right << "TangSubItr" << std::setw(15)
            << std::right << "norm(Res)" << std::setw(12) << std::right << "TangSubExit" << std::setw(14) << std::right
            << "QuasiNormItr" << std::setw(15) << std::right << "norm(Res)" << std::setw(15) << std::right
            << "QuasiNormExit" << "\n" << std::flush;
}

void DOTk_InexactTrustRegionSqpIO::writeInitialDiagnostics(const dotk::DOTk_InexactTrustRegionSQP* const alg_,
                                                           const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    size_t num_opt_itr_done = alg_->getNumItrDone();
    size_t num_trust_region_subproblem_itr = alg_->getNumTrustRegionSubProblemItrDone();
    size_t objective_function_counter = mng_->getObjectiveFunctionEvaluationCounter();

    Real new_objective_function_value = mng_->getNewObjectiveFunctionValue();
    Real norm_grad = mng_->getNewGradient()->norm();

    Real norm_trial_step = mng_->getTrialStep()->norm();

    Real norm_eq_constraint_residual = mng_->getNewEqualityConstraintResidual()->norm();

    this->writeHeader();
    m_DiagnosticsFile << std::setw(6) << std::right << std::scientific << std::setprecision(3) << num_opt_itr_done
            << std::setw(12) << std::right << objective_function_counter << std::setw(12) << std::right
            << new_objective_function_value << std::setw(12) << std::right << norm_grad << std::setw(12) << std::right
            << norm_eq_constraint_residual << std::setw(12) << std::right << norm_trial_step << std::setw(9)
            << std::right << "*" << std::setw(12) << std::right << "*" << std::setw(13) << std::right << "*"
            << std::setw(13) << std::right << "*" << std::setw(12) << std::right << "*" << std::setw(10) << std::right
            << "*" << std::setw(13) << std::right << "*" << std::setw(15) << std::right << "*" << std::setw(14)
            << std::right << "*" << std::setw(15) << std::right << "*" << std::setw(12) << std::right << "*"
            << std::setw(14) << std::right << "*" << std::setw(15) << std::right << "*" << std::setw(12) << std::right
            << "*" << std::setw(14) << std::right << "*" << std::setw(15) << std::right << "*" << std::setw(15)
            << std::right << "*" << "\n" << std::flush;
}

void DOTk_InexactTrustRegionSqpIO::writeFirstTrustRegionSubProblemItrDiagnostics(const dotk::DOTk_InexactTrustRegionSQP* const alg_,
                                                                                 const std::tr1::shared_ptr<
                                                                                         dotk::DOTk_TrustRegionMngTypeELP> & mng_,
                                                                                 const std::tr1::shared_ptr<
                                                                                         dotk::DOTk_InexactTrustRegionSqpSolverMng> & solver_mng_)
{
    size_t num_opt_itr_done = alg_->getNumItrDone();
    size_t num_trust_region_subproblem_itr = alg_->getNumTrustRegionSubProblemItrDone() + 1;
    size_t objective_function_counter = mng_->getObjectiveFunctionEvaluationCounter();

    Real new_objective_function_value = mng_->getNewObjectiveFunctionValue();
    Real norm_grad = mng_->getNewGradient()->norm();

    Real norm_trial_step = mng_->getTrialStep()->norm();

    Real norm_eq_constraint_residual = mng_->getNewEqualityConstraintResidual()->norm();

    std::ostringstream dual_prob_exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(alg_->getDualProbExitCriterion(), dual_prob_exit_criterion);
    std::ostringstream quasi_normal_prob_exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(alg_->getNormalProbExitCriterion(), quasi_normal_prob_exit_criterion);
    std::ostringstream tangential_prob_exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(alg_->getTangentialProbExitCriterion(), tangential_prob_exit_criterion);
    std::ostringstream tangential_subprob_exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(alg_->getTangentialSubProbExitCriterion(), tangential_subprob_exit_criterion);

    m_DiagnosticsFile << std::setw(6) << std::right << std::scientific << std::setprecision(3) << num_opt_itr_done
            << std::setw(12) << std::right << objective_function_counter << std::setw(12) << std::right
            << new_objective_function_value << std::setw(12) << std::right << norm_grad << std::setw(12) << std::right
            << norm_eq_constraint_residual << std::setw(12) << std::right << norm_trial_step << std::setw(9)
            << std::right << alg_->getNumTrustRegionSubProblemItrDone() << std::setw(12) << std::right
            << mng_->getTrustRegionRadius() << std::setw(13) << std::right << alg_->getActualReduction()
            << std::setw(13) << std::right << alg_->getPredictedReduction() << std::setw(12) << std::right
            << alg_->getActualOverPredictedReductionRatio() << std::setw(10) << std::right
            << solver_mng_->getNumTangentialProbItrDone() << std::setw(13) << std::right
            << solver_mng_->getTangentialProbResidualNorm() << std::setw(15) << std::right
            << tangential_prob_exit_criterion.str().c_str() << std::setw(14) << std::right
            << solver_mng_->getNumDualProbItrDone() << std::setw(15) << std::right
            << solver_mng_->getDualProbResidualNorm() << std::setw(12) << std::right
            << dual_prob_exit_criterion.str().c_str() << std::setw(14) << std::right
            << solver_mng_->getNumQuasiNormalProbItrDone() << std::setw(15) << std::right
            << solver_mng_->getQuasiNormalProbResidualNorm() << std::setw(12) << std::right
            << quasi_normal_prob_exit_criterion.str().c_str() << std::setw(14) << std::right
            << solver_mng_->getNumTangentialProbItrDone() << std::setw(15) << std::right
            << solver_mng_->getTangentialSubProbResidualNorm() << std::setw(15) << std::right
            << tangential_subprob_exit_criterion.str().c_str() << "\n" << std::flush;
}

void DOTk_InexactTrustRegionSqpIO::writeTrustRegionSubProblemDiagnostics(const dotk::DOTk_InexactTrustRegionSQP* const alg_,
                                                                         const std::tr1::shared_ptr<
                                                                                 dotk::DOTk_TrustRegionMngTypeELP> & mng_,
                                                                         const std::tr1::shared_ptr<
                                                                                 dotk::DOTk_InexactTrustRegionSqpSolverMng> & solver_mng_)
{
    size_t num_opt_itr_done = alg_->getNumItrDone();
    size_t num_trust_region_subproblem_itr = alg_->getNumTrustRegionSubProblemItrDone();
    size_t objective_function_counter = mng_->getObjectiveFunctionEvaluationCounter();

    Real new_objective_function_value = mng_->getNewObjectiveFunctionValue();
    Real norm_grad = mng_->getNewGradient()->norm();

    Real norm_trial_step = mng_->getTrialStep()->norm();

    Real norm_eq_constraint_residual = mng_->getNewEqualityConstraintResidual()->norm();

    std::ostringstream dual_prob_exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(alg_->getDualProbExitCriterion(), dual_prob_exit_criterion);
    std::ostringstream quasi_normal_prob_exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(alg_->getNormalProbExitCriterion(), quasi_normal_prob_exit_criterion);
    std::ostringstream tangential_prob_exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(alg_->getTangentialProbExitCriterion(), tangential_prob_exit_criterion);
    std::ostringstream tangential_subprob_exit_criterion;
    dotk::ioUtils::getSolverExitCriterion(alg_->getTangentialSubProbExitCriterion(), tangential_subprob_exit_criterion);

    m_DiagnosticsFile << std::setw(6) << std::right << std::scientific << std::setprecision(3) << "*" << std::setw(12)
            << std::right << "*" << std::setw(12) << std::right << "*" << std::setw(12) << std::right << "*"
            << std::setw(12) << std::right << "*" << std::setw(12) << std::right << "*" << std::setw(9) << std::right
            << alg_->getNumTrustRegionSubProblemItrDone() << std::setw(12) << std::right << mng_->getTrustRegionRadius()
            << std::setw(13) << std::right << alg_->getActualReduction() << std::setw(13) << std::right
            << alg_->getPredictedReduction() << std::setw(12) << std::right
            << alg_->getActualOverPredictedReductionRatio() << std::setw(10) << std::right
            << solver_mng_->getNumTangentialProbItrDone() << std::setw(13) << std::right
            << solver_mng_->getTangentialProbResidualNorm() << std::setw(15) << std::right
            << tangential_prob_exit_criterion.str().c_str() << std::setw(14) << std::right
            << solver_mng_->getNumDualProbItrDone() << std::setw(15) << std::right
            << solver_mng_->getDualProbResidualNorm() << std::setw(12) << std::right
            << dual_prob_exit_criterion.str().c_str() << std::setw(14) << std::right
            << solver_mng_->getNumQuasiNormalProbItrDone() << std::setw(15) << std::right
            << solver_mng_->getQuasiNormalProbResidualNorm() << std::setw(12) << std::right
            << quasi_normal_prob_exit_criterion.str().c_str() << std::setw(14) << std::right
            << solver_mng_->getNumTangentialProbItrDone() << std::setw(15) << std::right
            << solver_mng_->getTangentialSubProbResidualNorm() << std::setw(15) << std::right
            << tangential_subprob_exit_criterion.str().c_str() << "\n" << std::flush;
}

}
