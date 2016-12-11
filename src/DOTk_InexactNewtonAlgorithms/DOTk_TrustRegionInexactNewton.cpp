/*
 * DOTk_TrustRegionInexactNewton.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <fstream>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_TrustRegion.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_KrylovSolverFactory.hpp"
#include "DOTk_DescentDirectionTools.hpp"
#include "DOTk_TrustRegionInexactNewton.hpp"
#include "DOTk_TrustRegionInexactNewtonIO.hpp"
#include "DOTk_TrustRegionAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_TrustRegionInexactNewton::DOTk_TrustRegionInexactNewton(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
                                                             const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_) :
    dotk::DOTk_InexactNewtonAlgorithms(dotk::types::TRUST_REGION_INEXACT_NEWTON),
    m_NewObjectiveFuncValue(0.),
    m_KrylovSolver(),
    m_IO(new dotk::DOTk_TrustRegionInexactNewtonIO),
    m_LinearOperator(hessian_),
    m_DataMng(mng_),
    m_WorkVector(mng_->getTrialStep()->clone()),
    m_HessTimesTrialStep(mng_->getTrialStep()->clone())
{
}

DOTk_TrustRegionInexactNewton::DOTk_TrustRegionInexactNewton(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
                                                             const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_,
                                                             const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & solver_mng_) :
    dotk::DOTk_InexactNewtonAlgorithms(dotk::types::TRUST_REGION_INEXACT_NEWTON),
    m_NewObjectiveFuncValue(0.),
    m_KrylovSolver(),
    m_IO(new dotk::DOTk_TrustRegionInexactNewtonIO),
    m_LinearOperator(hessian_),
    m_DataMng(mng_),
    m_WorkVector(mng_->getTrialStep()->clone()),
    m_HessTimesTrialStep(mng_->getTrialStep()->clone())
{
    dotk::DOTk_KrylovSolverFactory krylov_solver_factory(solver_mng_->getSolverType());
    krylov_solver_factory.build(solver_mng_, m_KrylovSolver);
}

DOTk_TrustRegionInexactNewton::~DOTk_TrustRegionInexactNewton()
{
}

void DOTk_TrustRegionInexactNewton::setNewObjectiveFunctionValue(Real value_)
{
    m_NewObjectiveFuncValue = value_;
}

Real DOTk_TrustRegionInexactNewton::getNewObjectiveFunctionValue() const
{
    return (m_NewObjectiveFuncValue);
}

void DOTk_TrustRegionInexactNewton::setNumItrDone(size_t itr_)
{
    dotk::DOTk_InexactNewtonAlgorithms::setNumItrDone(itr_);
    m_LinearOperator->setNumOtimizationItrDone(itr_);
    m_KrylovSolver->getDataMng()->getLeftPrec()->setNumOptimizationItrDone(itr_);
}

void DOTk_TrustRegionInexactNewton::setMaxNumKrylovSolverItr(size_t itr_)
{
    m_KrylovSolver->setMaxNumKrylovSolverItr(itr_);
}

void DOTk_TrustRegionInexactNewton::setPrecGmresKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                            size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildPrecGmresSolver(primal_, m_LinearOperator, max_num_itr_, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(max_num_itr_);
}

void DOTk_TrustRegionInexactNewton::setLeftPrecCgKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                             size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgSolver(primal_, m_LinearOperator, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(max_num_itr_);
}

void DOTk_TrustRegionInexactNewton::setLeftPrecCrKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                             size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCrSolver(primal_, m_LinearOperator, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(max_num_itr_);
}

void DOTk_TrustRegionInexactNewton::setLeftPrecGcrKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                              size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecGcrSolver(primal_, m_LinearOperator, max_num_itr_, m_KrylovSolver);
}

void DOTk_TrustRegionInexactNewton::setLeftPrecCgneKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                               size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgneSolver(primal_, m_LinearOperator, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(max_num_itr_);
}

void DOTk_TrustRegionInexactNewton::setLeftPrecCgnrKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                               size_t max_num_itr_)
{
    dotk::DOTk_KrylovSolverFactory factory;
    factory.buildLeftPrecCgnrSolver(primal_, m_LinearOperator, m_KrylovSolver);
    m_KrylovSolver->setMaxNumKrylovSolverItr(max_num_itr_);
}

void DOTk_TrustRegionInexactNewton::printDiagnosticsAndSolutionEveryItr()
{
    m_IO->display(dotk::types::ITERATION);
}

void DOTk_TrustRegionInexactNewton::printDiagnosticsEveryItrAndSolutionAtTheEnd()
{
    m_IO->display(dotk::types::FINAL);
}

void DOTk_TrustRegionInexactNewton::getMin()
{
    m_IO->license();
    m_IO->openFile("DOTk_TrustRegionNewtonCGDiagnostics.out");
    this->checkAlgorithmInputs();

    this->initialize();
    m_KrylovSolver->setTrustRegionRadius(m_DataMng->getTrustRegionRadius());

    size_t iteration = 0;
    m_IO->printDiagnosticReport(this, m_KrylovSolver, m_DataMng);
    while(1)
    {
        ++iteration;
        this->setNumItrDone(iteration);
        m_LinearOperator->apply(m_DataMng, m_DataMng->getNewGradient(), m_DataMng->getMatrixTimesVector());

        this->solveTrustRegionSubProblem();

        m_LinearOperator->updateLimitedMemoryStorage(true);
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

void DOTk_TrustRegionInexactNewton::initialize()
{
    Real current_objective_function_value = m_DataMng->evaluateObjective();
    m_DataMng->setNewObjectiveFunctionValue(current_objective_function_value);
    m_DataMng->setOldObjectiveFunctionValue(current_objective_function_value);

    m_DataMng->computeGradient();
    Real initial_norm_gradient = m_DataMng->getNewGradient()->norm();
    m_DataMng->setNormNewGradient(initial_norm_gradient);
    m_DataMng->getOldPrimal()->update(1., *m_DataMng->getNewPrimal(), 0.);
    m_DataMng->getOldGradient()->update(1., *m_DataMng->getNewGradient(), 0.);
}

void DOTk_TrustRegionInexactNewton::checkAlgorithmInputs()
{
    std::ostringstream msg;
    m_DataMng->checkTrustRegionPtr(msg);
    if(m_KrylovSolver.use_count() == 0)
    {
        std::perror("\n**** Error in DOTk_TrustRegionInexactNewton::checkAlgorithmInputs -> User did not define Krylov solver. ABORT. ****\n");
        std::abort();
    }
}

void DOTk_TrustRegionInexactNewton::solveTrustRegionSubProblem()
{
    size_t trust_region_sub_problem_itr = 1;
    while(1)
    {
        m_DataMng->getTrustRegion()->setNumTrustRegionSubProblemItrDone(trust_region_sub_problem_itr);

        dotk::gtools::getSteepestDescent(m_DataMng->getNewGradient(), m_WorkVector);
        m_KrylovSolver->solve(m_WorkVector, m_Criterion, m_DataMng);

        dotk::DOTk_InexactNewtonAlgorithms::setTrialStep(m_KrylovSolver, m_DataMng);
        dotk::gtools::checkDescentDirection(m_DataMng->getNewGradient(),
                                            m_DataMng->getTrialStep(),
                                            dotk::DOTk_InexactNewtonAlgorithms::getMinCosineAngleTol());

        if(this->checkTrustRegionSubProblemConvergence() == true)
        {
            bool trust_region_sub_problem_converged = true;
            m_DataMng->updateState(this->getNewObjectiveFunctionValue(), m_WorkVector);
            m_IO->printDiagnosticReport(this, m_KrylovSolver, m_DataMng, trust_region_sub_problem_converged);
            break;
        }

        m_IO->printDiagnosticReport(this, m_KrylovSolver, m_DataMng);
        ++trust_region_sub_problem_itr;
    }
}

void DOTk_TrustRegionInexactNewton::computeScaledInexactNewtonStep()
{
    bool invalid_curvature_detected = m_KrylovSolver->invalidCurvatureWasDetected();
    dotk::types::trustregion_t trust_region_type = m_DataMng->getTrustRegion()->getTrustRegionType();
    switch(trust_region_type)
    {
        case dotk::types::TRUST_REGION_DISABLED:
        {
            break;
        }
        case dotk::types::TRUST_REGION_DOGLEG:
        {
            m_DataMng->computeScaledInexactNewtonStep(invalid_curvature_detected,
                                                      m_KrylovSolver->getDescentDirection());
            break;
        }
        case dotk::types::TRUST_REGION_CAUCHY:
        case dotk::types::TRUST_REGION_DOUBLE_DOGLEG:
        {
            m_DataMng->computeScaledInexactNewtonStep(invalid_curvature_detected, m_DataMng->getMatrixTimesVector());
            break;
        }
    }
}

void DOTk_TrustRegionInexactNewton::computeActualReduction()
{
    const std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_ptr = m_DataMng->getTrustRegion();
    m_WorkVector->update(1., *m_DataMng->getNewPrimal(), 0.);
    m_WorkVector->update(static_cast<Real>(1.0), *m_DataMng->getTrialStep(), 1.);

    Real new_objective_func_val = m_DataMng->evaluateObjective(m_WorkVector);
    this->setNewObjectiveFunctionValue(new_objective_func_val);

    Real old_objective_func_val = m_DataMng->getNewObjectiveFunctionValue();
    trust_region_ptr->computeActualReduction(new_objective_func_val, old_objective_func_val);
}

bool DOTk_TrustRegionInexactNewton::checkTrustRegionSubProblemConvergence()
{
    this->computeScaledInexactNewtonStep();
    bool trust_region_sub_problem_converged = false;
    const std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_ptr = m_DataMng->getTrustRegion();
    size_t trust_region_sub_problem_itr = trust_region_ptr->getNumTrustRegionSubProblemItrDone();

    this->computeActualReduction();
    m_KrylovSolver->getLinearOperator()->apply(m_DataMng, m_DataMng->getTrialStep(), m_HessTimesTrialStep);
    trust_region_ptr->computePredictedReduction(m_DataMng->getNewGradient(),
                                                m_DataMng->getTrialStep(),
                                                m_HessTimesTrialStep);

    Real norm_trial_step = m_DataMng->getTrialStep()->norm();
    size_t max_trust_region_itr = trust_region_ptr->getMaxTrustRegionSubProblemIterations();
    bool accept_trust_region = trust_region_ptr->acceptTrustRegionRadius(m_DataMng->getTrialStep());
    m_KrylovSolver->setTrustRegionRadius(trust_region_ptr->getTrustRegionRadius());

    if(accept_trust_region == true)
    {
        trust_region_sub_problem_converged = true;
    }
    else if(norm_trial_step < dotk::DOTk_InexactNewtonAlgorithms::getTrialStepTol())
    {
        trust_region_sub_problem_converged = true;
    }
    else if(trust_region_sub_problem_itr == max_trust_region_itr)
    {
        Real trust_region_radius = trust_region_ptr->getContractionParameter()
                                   * trust_region_ptr->getTrustRegionRadius();
        trust_region_ptr->setTrustRegionRadius(trust_region_radius);
        trust_region_sub_problem_converged = true;
    }
    return (trust_region_sub_problem_converged);
}

}
