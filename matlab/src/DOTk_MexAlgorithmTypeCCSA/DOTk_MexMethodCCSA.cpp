/*
 * DOTk_MexMethodCCSA.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexVector.hpp"
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
        m_DualSolverType(dotk::ccsa::dual_solver_t::NONLINEAR_CG),
        m_DualSolverTypeNLCG(dotk::types::nonlinearcg_t::POLAK_RIBIERE_NLCG)
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
    primal_solver_.setFeasibilityTolerance(m_FeasibilityTolerance);
    primal_solver_.setControlStagnationTolerance(m_ControlStagnationTolerance);

    primal_solver_.setDualObjectiveEpsilonParameter(m_DualObjectiveEpsilonParameter);
    primal_solver_.setMovingAsymptoteLowerBoundScale(m_MovingAsymptoteLowerBoundScale);
    primal_solver_.setMovingAsymptoteUpperBoundScale(m_MovingAsymptoteUpperBoundScale);
    primal_solver_.setMovingAsymptoteExpansionParameter(m_MovingAsymptoteExpansionParameter);
    primal_solver_.setMovingAsymptoteContractionParameter(m_MovingAsymptoteContractionParameter);
    primal_solver_.setDualObjectiveTrialControlBoundScaling(m_DualObjectiveTrialControlBoundScaling);
}

void DOTk_MexMethodCCSA::setDualSolverParameters(const std::shared_ptr<dotk::DOTk_DualSolverNLCG> & dual_solver_)
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
                                          const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_,
                                          mxArray* outputs_[])
{
    // Create memory allocation for output struc
    const char *field_names[9] =
        { "Iterations", "ObjectiveFunctionValue", "Control", "ObjectiveGradient", "NormObjectiveGradient",
                "InequalityResiduals", "Norm_KKT_Residual", "MaxFeasibilityMeasure", "ControlStagnationMeasure" };
    outputs_[0] = mxCreateStructMatrix(1, 1, 9, field_names);

    /* NOTE: mxSetField does not create a copy of the data allocated. Thus,
     * mxDestroyArray cannot be called. Furthermore, MEX array data (e.g.
     * control, gradient, etc.) should be duplicated since the data in the
     * manager will be deallocated at the end. */
    mxArray* mx_number_iterations = mxCreateNumericMatrix(1, 1, mxINDEX_CLASS, mxREAL);
    static_cast<size_t*>(mxGetData(mx_number_iterations))[0] = algorithm_.getIterationCount();
    mxSetField(outputs_[0], 0, "Iterations", mx_number_iterations);

    double value = data_mng_->m_CurrentObjectiveFunctionValue;
    mxArray* mx_objective_function_value = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "ObjectiveFunctionValue", mx_objective_function_value);

    dotk::MexVector & control = dynamic_cast<dotk::MexVector &>(*data_mng_->m_CurrentControl);
    mxArray* mx_control = mxDuplicateArray(control.array());
    mxSetField(outputs_[0], 0, "Control", mx_control);

    dotk::MexVector & gradient = dynamic_cast<dotk::MexVector &>(*data_mng_->m_CurrentObjectiveGradient);
    mxArray* mx_gradient = mxDuplicateArray(gradient.array());
    mxSetField(outputs_[0], 0, "ObjectiveGradient", mx_gradient);

    value = gradient.norm();
    mxArray* mx_norm_gradient = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormObjectiveGradient", mx_norm_gradient);

    dotk::MexVector & inequality_residuals = dynamic_cast<dotk::MexVector &>(*data_mng_->m_CurrentInequalityResiduals);
    mxArray* mx_inequality_residuals = mxDuplicateArray(inequality_residuals.array());
    mxSetField(outputs_[0], 0, "InequalityResiduals", mx_inequality_residuals);

    value = algorithm_.getCurrentResidualNorm();
    mxArray* mx_norm_kkt_residual = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "Norm_KKT_Residual", mx_norm_kkt_residual);

    value = algorithm_.getCurrentMaxFeasibilityMeasure();
    mxArray* mx_max_feasibility_measure = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "MaxFeasibilityMeasure", mx_max_feasibility_measure);

    value = algorithm_.getCurrentControlStagnationMeasure();
    mxArray* mx_control_stagnation_measure = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "ControlStagnationMeasure", mx_control_stagnation_measure);
}

void DOTk_MexMethodCCSA::initialize(const mxArray* options_)
{
    m_ProblemType = dotk::mex::parseProblemType(options_);
    m_DualSolverType = dotk::mex::parseDualSolverType(options_);
    m_DualSolverTypeNLCG = dotk::mex::parseDualSolverTypeNLCG(options_);

    m_PrimalSolverMaxNumberIterations = dotk::mex::parseMaxNumOuterIterations(options_);
    m_DualSolverMaxNumberLineSearchIterations = dotk::mex::parseMaxNumLineSearchItr(options_);
    m_DualSolverMaxNumberIterations = dotk::mex::parseDualSolverMaxNumberIterations(options_);

    m_GradientTolerance = dotk::mex::parseGradientTolerance(options_);
    m_ResidualTolerance = dotk::mex::parseResidualTolerance(options_);
    m_FeasibilityTolerance = dotk::mex::parseFeasibilityTolerance(options_);
    m_ControlStagnationTolerance = dotk::mex::parseControlStagnationTolerance(options_);

    m_MovingAsymptoteUpperBoundScale = dotk::mex::parseMovingAsymptoteUpperBoundScale(options_);
    m_MovingAsymptoteLowerBoundScale = dotk::mex::parseMovingAsymptoteLowerBoundScale(options_);
    m_MovingAsymptoteExpansionParameter = dotk::mex::parseMovingAsymptoteExpansionParameter(options_);
    m_MovingAsymptoteContractionParameter = dotk::mex::parseMovingAsymptoteContractionParameter(options_);

    m_DualSolverTrialStepTolerance = dotk::mex::parseDualSolverStepTolerance(options_);
    m_DualSolverGradientTolerance = dotk::mex::parseDualSolverGradientTolerance(options_);
    m_DualSolverLineSearchStepUpperBound = dotk::mex::parseLineSearchStepUpperBound(options_);
    m_DualSolverLineSearchStepLowerBound = dotk::mex::parseLineSearchStepLowerBound(options_);
    m_DualObjectiveEpsilonParameter = dotk::mex::parseDualObjectiveRelaxationParameter(options_);
    m_DualObjectiveTrialControlBoundScaling = dotk::mex::parseDualObjectiveControlBoundsScaling(options_);
    m_DualSolverObjectiveStagnationTolerance = dotk::mex::parseDualSolverObjectiveStagnationTolerance(options_);
}

}
