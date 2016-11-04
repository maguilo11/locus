/*
 * DOTk_MexMethodCCSA.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_DataMngCCSA.hpp"
#include "DOTk_MexMethodCCSA.hpp"
#include "DOTk_AlgorithmCCSA.hpp"
#include "DOTk_DualSolverNLCG.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexMethodCcsaParser.hpp"

namespace dotk
{

DOTk_MexMethodCCSA::DOTk_MexMethodCCSA(const mxArray* options_) :
        m_DualSolverMaxNumberIterations(10),
        m_PrimalSolverMaxNumberIterations(50),
        m_DualSolverMaxNumberLineSearchIterations(5),
        m_GradientTolerance(1e-3),
        m_ResidualTolerance(1e-4),
        m_OptimalityTolerance(1e-3),
        m_FeasibilityTolerance(1e-4),
        m_ControlStagnationTolerance(1e-3),
        m_MovingAsymptoteUpperBoundScale(10),
        m_MovingAsymptoteLowerBoundScale(0.1),
        m_MovingAsymptoteExpansionParameter(1.2),
        m_MovingAsymptoteContractionParameter(0.4),
        m_DualSolverGradientTolerance(1e-8),
        m_DualSolverTrialStepTolerance(1e-8),
        m_DualObjectiveEpsilonParameter(1e-6),
        m_DualSolverLineSearchStepLowerBound(1e-3),
        m_DualSolverLineSearchStepUpperBound(0.5),
        m_DualObjectiveTrialControlBoundScaling(0.5),
        m_DualSolverObjectiveStagnationTolerance(1e-8),
        m_ProblemType(dotk::types::problem_t::PROBLEM_TYPE_UNDEFINED),
        m_DualSolverTypeNLCG(dotk::types::nonlinearcg_t::POLAK_RIBIERE_NLCG),
        m_DualSolverType(dotk::ccsa::dual_solver_t::NONLINEAR_CG)
{
    this->initialize(options_);
}

DOTk_MexMethodCCSA::~DOTk_MexMethodCCSA()
{
}

dotk::types::problem_t DOTk_MexMethodCCSA::getProblemType() const
{
    return (m_ProblemType);
}

dotk::ccsa::dual_solver_t DOTk_MexMethodCCSA::getDualSolverType() const
{
    return (m_DualSolverType);
}

dotk::types::nonlinearcg_t DOTk_MexMethodCCSA::getNonlinearConjugateGradientType() const
{
    return (m_DualSolverTypeNLCG);
}

void DOTk_MexMethodCCSA::setPrimalSolverParameters(dotk::DOTk_AlgorithmCCSA & primal_solver_)
{
    primal_solver_.setMaxNumIterations(m_PrimalSolverMaxNumberIterations);

    primal_solver_.setResidualTolerance(m_ResidualTolerance);
    primal_solver_.setGradientTolerance(m_GradientTolerance);
    primal_solver_.setOptimalityTolerance(m_OptimalityTolerance);
    primal_solver_.setFeasibilityTolerance(m_FeasibilityTolerance);
    primal_solver_.setControlStagnationTolerance(m_ControlStagnationTolerance);

    primal_solver_.setDualObjectiveEpsilonParameter(m_DualObjectiveEpsilonParameter);
    primal_solver_.setMovingAsymptoteLowerBoundScale(m_MovingAsymptoteLowerBoundScale);
    primal_solver_.setMovingAsymptoteUpperBoundScale(m_MovingAsymptoteUpperBoundScale);
    primal_solver_.setMovingAsymptoteExpansionParameter(m_MovingAsymptoteExpansionParameter);
    primal_solver_.setMovingAsymptoteContractionParameter(m_MovingAsymptoteContractionParameter);
    primal_solver_.setDualObjectiveTrialControlBoundScaling(m_DualObjectiveTrialControlBoundScaling);
}

void DOTk_MexMethodCCSA::setDualSolverParameters(const std::tr1::shared_ptr<dotk::DOTk_DualSolverNLCG> & dual_solver_)
{
    dual_solver_->setNonlinearCgType(m_DualSolverTypeNLCG);
    dual_solver_->setMaxNumIterations(m_DualSolverMaxNumberIterations);
    dual_solver_->setGradientTolerance(m_DualSolverGradientTolerance);
    dual_solver_->setTrialStepTolerance(m_DualSolverTrialStepTolerance);
    dual_solver_->setLineSearchStepLowerBound(m_DualSolverLineSearchStepLowerBound);
    dual_solver_->setLineSearchStepUpperBound(m_DualSolverLineSearchStepUpperBound);
    dual_solver_->setMaxNumLineSearchIterations(m_DualSolverMaxNumberLineSearchIterations);
    dual_solver_->setObjectiveStagnationTolerance(m_DualSolverObjectiveStagnationTolerance);
}

void DOTk_MexMethodCCSA::gatherOutputData(dotk::DOTk_AlgorithmCCSA & algorithm_,
                                          const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_,
                                          mxArray* output_[])
{
    // Create memory allocation for output struc
    const char *field_names[10] =
        { "Iterations", "ObjectiveFunctionValue", "Control", "ObjectiveGradient", "NormObjectiveGradient",
                "InequalityConstraintResiduals", "Dual", "NormResidualKKT", "MaxFeasibilityMeasure",
                "ControlStagnationMeasure" };
    output_[0] = mxCreateStructMatrix(1, 1, 10, field_names);

    dotk::DOTk_MexArrayPtr iterations(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(iterations.get()))[0] = algorithm_.getIterationCount();
    mxSetField(output_[0], 0, "Iterations", iterations.get());

    dotk::DOTk_MexArrayPtr objective(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(objective.get())[0] = data_mng_->m_CurrentObjectiveFunctionValue;
    mxSetField(output_[0], 0, "ObjectiveFunctionValue", objective.get());

    size_t number_controls = data_mng_->m_CurrentControl->size();
    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(number_controls, 1, mxREAL));
    data_mng_->m_CurrentControl->gather(mxGetPr(control.get()));
    mxSetField(output_[0], 0, "Control", control.get());

    dotk::DOTk_MexArrayPtr gradient(mxCreateDoubleMatrix(number_controls, 1, mxREAL));
    data_mng_->m_CurrentObjectiveGradient->gather(mxGetPr(gradient.get()));
    mxSetField(output_[0], 0, "ObjectiveGradient", gradient.get());

    dotk::DOTk_MexArrayPtr norm_pruned_gradient(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_pruned_gradient.get())[0] = algorithm_.getCurrentObjectiveGradientNorm();
    mxSetField(output_[0], 0, "NormObjectiveGradient", norm_pruned_gradient.get());

    size_t number_duals = data_mng_->m_Dual->size();
    dotk::DOTk_MexArrayPtr residuals(mxCreateDoubleMatrix(number_duals, 1, mxREAL));
    data_mng_->m_CurrentInequalityResiduals->gather(mxGetPr(residuals.get()));
    mxSetField(output_[0], 0, "InequalityConstraintResiduals", residuals.get());

    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(number_duals, 1, mxREAL));
    data_mng_->m_Dual->gather(mxGetPr(dual.get()));
    mxSetField(output_[0], 0, "Dual", dual.get());

    dotk::DOTk_MexArrayPtr norm_kkt(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_kkt.get())[0] = algorithm_.getCurrentResidualNorm();
    mxSetField(output_[0], 0, "NormResidualKKT", norm_kkt.get());

    dotk::DOTk_MexArrayPtr max_feasibility_measure(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(max_feasibility_measure.get())[0] = algorithm_.getCurrentMaxFeasibilityMeasure();
    mxSetField(output_[0], 0, "MaxFeasibilityMeasure", max_feasibility_measure.get());

    dotk::DOTk_MexArrayPtr control_stagnation_measure(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(control_stagnation_measure.get())[0] = algorithm_.getCurrentControlStagnationMeasure();
    mxSetField(output_[0], 0, "ControlStagnationMeasure", control_stagnation_measure.get());

    iterations.release();
    objective.release();
    control.release();
    gradient.release();
    norm_pruned_gradient.release();
    residuals.release();
    dual.release();
    norm_kkt.release();
    max_feasibility_measure.release();
    control_stagnation_measure.release();
}

void DOTk_MexMethodCCSA::initialize(const mxArray* options_)
{
    dotk::mex::parseProblemType(options_, m_ProblemType);
    dotk::mex::parseDualSolverType(options_, m_DualSolverType);
    dotk::mex::parseDualSolverTypeNLCG(options_, m_DualSolverTypeNLCG);

    dotk::mex::parseMaxNumAlgorithmItr(options_, m_PrimalSolverMaxNumberIterations);
    dotk::mex::parseMaxNumLineSearchItr(options_, m_DualSolverMaxNumberLineSearchIterations);
    dotk::mex::parseDualSolverMaxNumberIterations(options_, m_DualSolverMaxNumberIterations);

    dotk::mex::parseGradientTolerance(options_, m_GradientTolerance);
    dotk::mex::parseResidualTolerance(options_, m_ResidualTolerance);
    dotk::mex::parseOptimalityTolerance(options_, m_OptimalityTolerance);
    dotk::mex::parseFeasibilityTolerance(options_, m_FeasibilityTolerance);
    dotk::mex::parseControlStagnationTolerance(options_, m_ControlStagnationTolerance);

    dotk::mex::parseMovingAsymptoteUpperBoundScale(options_, m_MovingAsymptoteUpperBoundScale);
    dotk::mex::parseMovingAsymptoteLowerBoundScale(options_, m_MovingAsymptoteLowerBoundScale);
    dotk::mex::parseMovingAsymptoteExpansionParameter(options_, m_MovingAsymptoteExpansionParameter);
    dotk::mex::parseMovingAsymptoteContractionParameter(options_, m_MovingAsymptoteContractionParameter);

    dotk::mex::parseLineSearchStepLowerBound(options_, m_DualSolverLineSearchStepLowerBound);
    dotk::mex::parseLineSearchStepUpperBound(options_, m_DualSolverLineSearchStepUpperBound);
    dotk::mex::parseDualSolverGradientTolerance(options_, m_DualSolverGradientTolerance);
    dotk::mex::parseDualSolverTrialStepTolerance(options_, m_DualSolverTrialStepTolerance);
    dotk::mex::parseDualObjectiveEpsilonParameter(options_, m_DualObjectiveEpsilonParameter);
    dotk::mex::parseDualObjectiveTrialControlBoundScaling(options_, m_DualObjectiveTrialControlBoundScaling);
    dotk::mex::parseDualSolverObjectiveStagnationTolerance(options_, m_DualSolverObjectiveStagnationTolerance);
}

}
