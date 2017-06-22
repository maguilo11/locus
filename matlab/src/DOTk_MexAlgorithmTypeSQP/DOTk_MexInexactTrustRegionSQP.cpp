/*
 * DOTk_MexInexactTrustRegionSQP.cpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexParseAlgorithmTypeSQP.hpp"
#include "DOTk_MexInexactTrustRegionSQP.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_InexactTrustRegionSQP.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"
#include "DOTk_InexactTrustRegionSqpSolverMng.hpp"

namespace dotk
{

DOTk_MexInexactTrustRegionSQP::DOTk_MexInexactTrustRegionSQP(const mxArray* options_[]) :
        dotk::DOTk_MexAlgorithmTypeSQP(options_[0]),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr)
{
    this->initialize(options_);
}

DOTk_MexInexactTrustRegionSQP::~DOTk_MexInexactTrustRegionSQP()
{
    this->clear();
}

void DOTk_MexInexactTrustRegionSQP::solve(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeSQP::getProblemType();

    switch (type)
    {
        case dotk::types::TYPE_ELP:
        {
            this->solveTypeEqualityConstrainedLinearProgramming(input_, output_);
            break;
        }
        case dotk::types::TYPE_ENLP:
        {
            this->solveTypeEqualityConstrainedNonlinearProgramming(input_, output_);
            break;
        }
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_NLP_BOUND:
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_UNLP:
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_CNLP:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::TYPE_ILP:
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::string msg(" DOTk/MEX ERROR: Invalid Problem Type for Inexact Trust Region Sequential Quadratic Programming (SQP) Method. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexInexactTrustRegionSQP::setTrustRegionParameters(const mxArray* options_,
                                                             dotk::DOTk_TrustRegionMngTypeELP & mng_)
{
    double value = dotk::mex::parseMaxTrustRegionRadius(options_);
    mng_.setMaxTrustRegionRadius(value);
    value = dotk::mex::parseMinTrustRegionRadius(options_);
    mng_.setMinTrustRegionRadius(value);
    value = dotk::mex::parseInitialTrustRegionRadius(options_);
    mng_.setTrustRegionRadius(value);
    value = dotk::mex::parseTrustRegionExpansionFactor(options_);
    mng_.setTrustRegionExpansionParameter(value);
    value = dotk::mex::parseTrustRegionContractionFactor(options_);
    mng_.setTrustRegionContractionParameter(value);
    value = dotk::mex::parseMinActualOverPredictedReductionRatio(options_);
    mng_.setMinActualOverPredictedReductionAllowed(value);
}

void DOTk_MexInexactTrustRegionSQP::setSqpKrylovSolversParameters(const mxArray* options_,
                                                                  dotk::DOTk_InexactTrustRegionSqpSolverMng & mng_)
{
    size_t value = dotk::mex::parseSqpMaxNumDualProblemItr(options_);
    mng_.setMaxNumDualProblemItr(value);
    value = dotk::mex::parseSqpMaxNumTangentialProblemItr(options_);
    mng_.setMaxNumTangentialProblemItr(value);
    value = dotk::mex::parseSqpMaxNumQuasiNormalProblemItr(options_);
    mng_.setMaxNumQuasiNormalProblemItr(value);
    value = dotk::mex::parseSqpMaxNumTangentialSubProblemItr(options_);
    mng_.setMaxNumTangentialSubProblemItr(value);
}

void DOTk_MexInexactTrustRegionSQP::setAlgorithmParameters(const mxArray* options_,
                                                           dotk::DOTk_InexactTrustRegionSQP & algorithm_)
{
    double value = dotk::mex::parseTangentialTolerance(options_);
    algorithm_.setTangentialTolerance(value);
    value = dotk::mex::parseDualProblemTolerance(options_);
    algorithm_.setDualProblemTolerance(value);
    value = dotk::mex::parseDualDotGradientTolerance(options_);
    algorithm_.setDualDotGradientTolerance(value);
    value = dotk::mex::parseToleranceContractionFactor(options_);
    algorithm_.setToleranceContractionFactor(value);
    value = dotk::mex::parsePredictedReductionParameter(options_);
    algorithm_.setPredictedReductionParameter(value);
    value = dotk::mex::parseMeritFunctionPenaltyParameter(options_);
    algorithm_.setMeritFunctionPenaltyParameter(value);
    value = dotk::mex::parseQuasiNormalProblemRelativeTolerance(options_);
    algorithm_.setQuasiNormalProblemRelativeTolerance(value);
    value = dotk::mex::parseTangentialToleranceContractionFactor(options_);
    algorithm_.setTangentialToleranceContractionFactor(value);
    value = dotk::mex::parseActualOverPredictedReductionTolerance(options_);
    algorithm_.setActualOverPredictedReductionTolerance(value);
    value = dotk::mex::parseMaxEffectiveTangentialOverTrialStepRatio(options_);
    algorithm_.setMaxEffectiveTangentialOverTrialStepRatio(value);
    value = dotk::mex::parseTangentialSubProbLeftPrecProjectionTolerance(options_);
    algorithm_.setTangentialSubProbLeftPrecProjectionTolerance(value);
    value = dotk::mex::parseQuasiNormalProblemTrustRegionRadiusPenaltyParameter(options_);
    algorithm_.setQuasiNormalProblemTrustRegionRadiusPenaltyParameter(value);
}

void DOTk_MexInexactTrustRegionSQP::solveTypeEqualityConstrainedLinearProgramming(const mxArray* input_[],
                                                                                  mxArray* output_[])
{
    // Set core data structures: dual and control vectors
    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector duals(num_duals, 0.);
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);

    // Allocate DOTk data structures
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateUserDefinedDual(duals);
    primal->allocateUserDefinedControl(controls);

    // Set objective function and equality constraint
    dotk::types::problem_t problem_type = DOTk_MexAlgorithmTypeSQP::getProblemType();
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, problem_type));
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality(new dotk::DOTk_MexEqualityConstraint(m_EqualityConstraint, problem_type));

    // Set trust region SQP data manager
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        data_mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    this->setTrustRegionParameters(input_[0], *data_mng);

    // Set solver manager
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, data_mng));
    this->setSqpKrylovSolversParameters(input_[0], *solver_mng);

    // Initialize SQP algorithm
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP algorithm(hessian, data_mng, solver_mng);
    this->setAlgorithmParameters(input_[0], algorithm);

    algorithm.getMin();

    this->gatherOutputDataTypeLP(algorithm, *data_mng, output_);
}

void DOTk_MexInexactTrustRegionSQP::solveTypeEqualityConstrainedNonlinearProgramming(const mxArray* input_[],
                                                                                     mxArray* output_[])
{
    // Set core data structures: dual, state, and control vectors
    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector duals(num_duals, 0.);
    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector states(num_states, 0.);
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);

    // Allocate DOTk data structures
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateUserDefinedDual(duals);
    primal->allocateUserDefinedState(states);
    primal->allocateUserDefinedControl(controls);

    // Set objective function and equality constraint
    dotk::types::problem_t problem_type = DOTk_MexAlgorithmTypeSQP::getProblemType();
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, problem_type));
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality(new dotk::DOTk_MexEqualityConstraint(m_EqualityConstraint, problem_type));

    // Set trust region SQP data manager
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        data_mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    this->setTrustRegionParameters(input_[0], *data_mng);

    // Set solver manager
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, data_mng));
    this->setSqpKrylovSolversParameters(input_[0], *solver_mng);

    // Initialize SQP algorithm
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP algorithm(hessian, data_mng, solver_mng);
    this->setAlgorithmParameters(input_[0], algorithm);

    algorithm.getMin();

    this->gatherOutputDataTypeNLP(algorithm, *data_mng, output_);
}

void DOTk_MexInexactTrustRegionSQP::gatherOutputDataTypeLP(dotk::DOTk_InexactTrustRegionSQP & algorithm_,
                                                           dotk::DOTk_TrustRegionMngTypeELP & mng_,
                                                           mxArray* outputs_[])
{
    // Create memory allocation for output struct
    const char *field_names[8] =
        { "Iterations", "ObjectiveFunctionValue", "Primal", "Dual", "Gradient", "NormGradient", "Step", "NormStep" };
    outputs_[0] = mxCreateStructMatrix(1, 1, 8, field_names);

    /* NOTE: mxSetField does not create a copy of the data allocated. Thus,
     * mxDestroyArray cannot be called. Furthermore, MEX array data (e.g.
     * control, gradient, etc.) should be duplicated since the data in the
     * manager will be deallocated at the end. */
    mxArray* number_iterations = mxCreateNumericMatrix(1, 1, mxINDEX_CLASS, mxREAL);
    static_cast<size_t*>(mxGetData(number_iterations))[0] = algorithm_.getNumItrDone();
    mxSetField(outputs_[0], 0, "Iterations", number_iterations);

    double value = mng_.getNewObjectiveFunctionValue();
    mxArray* mx_objective_function_value = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "ObjectiveFunctionValue", mx_objective_function_value);

    dotk::MexVector & control = dynamic_cast<dotk::MexVector &>(*mng_.getNewPrimal());
    mxArray* mx_control = mxDuplicateArray(control.array());
    mxSetField(outputs_[0], 0, "Control", mx_control);

    dotk::MexVector & dual = dynamic_cast<dotk::MexVector &>(*mng_.getNewDual());
    mxArray* mx_dual = mxDuplicateArray(dual.array());
    mxSetField(outputs_[0], 0, "Dual", mx_dual);

    dotk::MexVector & gradient = dynamic_cast<dotk::MexVector &>(*mng_.getNewGradient());
    mxArray* mx_gradient = mxDuplicateArray(gradient.array());
    mxSetField(outputs_[0], 0, "Gradient", mx_gradient);

    value = gradient.norm();
    mxArray* mx_norm_gradient = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormGradient", mx_norm_gradient);

    dotk::MexVector & step = dynamic_cast<dotk::MexVector &>(*mng_.getTrialStep());
    mxArray* mx_step = mxDuplicateArray(step.array());
    mxSetField(outputs_[0], 0, "Step", mx_step);

    value = step.norm();
    mxArray* mx_norm_step = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormStep", mx_norm_step);
}

void DOTk_MexInexactTrustRegionSQP::gatherOutputDataTypeNLP(dotk::DOTk_InexactTrustRegionSQP & algorithm_,
                                                            dotk::DOTk_TrustRegionMngTypeELP & mng_,
                                                            mxArray* outputs_[])
{
    // Create memory allocation for output struct
    const char *field_names[13] =
                { "Iterations", "ObjectiveFunctionValue", "State", "Control", "Dual", "StateGradient",
                        "NormStateGradient", "ControlGradient", "NormControlGradient", "StateStep", "NormStateStep",
                        "ControlStep", "NormControlStep" };
    outputs_[0] = mxCreateStructMatrix(1, 1, 13, field_names);

    /* NOTE: mxSetField does not create a copy of the data allocated. Thus,
     * mxDestroyArray cannot be called. Furthermore, MEX array data (e.g.
     * control, gradient, etc.) should be duplicated since the data in the
     * manager will be deallocated at the end. */
    mxArray* mx_number_iterations = mxCreateNumericMatrix(1, 1, mxINDEX_CLASS, mxREAL);
    static_cast<size_t*>(mxGetData(mx_number_iterations))[0] = algorithm_.getNumItrDone();
    mxSetField(outputs_[0], 0, "Iterations", mx_number_iterations);

    double value = mng_.getNewObjectiveFunctionValue();
    mxArray* mx_objective_function_value = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "ObjectiveFunctionValue", mx_objective_function_value);

    dotk::MexVector & state = dynamic_cast<dotk::MexVector &>(*mng_.getNewPrimal()->state());
    mxArray* mx_state = mxDuplicateArray(state.array());
    mxSetField(outputs_[0], 0, "State", mx_state);

    dotk::MexVector & control = dynamic_cast<dotk::MexVector &>(*mng_.getNewPrimal()->control());
    mxArray* mx_control = mxDuplicateArray(control.array());
    mxSetField(outputs_[0], 0, "Control", mx_control);

    dotk::MexVector & dual = dynamic_cast<dotk::MexVector &>(*mng_.getNewDual());
    mxArray* mx_dual = mxDuplicateArray(dual.array());
    mxSetField(outputs_[0], 0, "Dual", mx_dual);

    dotk::MexVector & state_gradient = dynamic_cast<dotk::MexVector &>(*mng_.getNewGradient()->state());
    mxArray* mx_state_gradient = mxDuplicateArray(state_gradient.array());
    mxSetField(outputs_[0], 0, "StateGradient", mx_state_gradient);

    value = state_gradient.norm();
    mxArray* mx_norm_state_gradient = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormStateGradient", mx_norm_state_gradient);

    dotk::MexVector & control_gradient = dynamic_cast<dotk::MexVector &>(*mng_.getNewGradient()->control());
    mxArray* mx_control_gradient = mxDuplicateArray(control_gradient.array());
    mxSetField(outputs_[0], 0, "ControlGradient", mx_control_gradient);

    value = control_gradient.norm();
    mxArray* mx_norm_control_gradient = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormControlGradient", mx_norm_control_gradient);

    dotk::MexVector & state_step = dynamic_cast<dotk::MexVector &>(*mng_.getTrialStep()->state());
    mxArray* mx_state_step = mxDuplicateArray(state_step.array());
    mxSetField(outputs_[0], 0, "StateStep", mx_state_step);

    value = state_step.norm();
    mxArray* mx_norm_state_step = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormStateStep", mx_norm_state_step);

    dotk::MexVector & control_step = dynamic_cast<dotk::MexVector &>(*mng_.getTrialStep()->control());
    mxArray* mx_control_step = mxDuplicateArray(control_step.array());
    mxSetField(outputs_[0], 0, "ControlStep", mx_control_step);

    value = control_step.norm();
    mxArray* mx_norm_control_step = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormControlStep", mx_norm_control_step);
}

void DOTk_MexInexactTrustRegionSQP::clear()
{
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
}

void DOTk_MexInexactTrustRegionSQP::initialize(const mxArray* options_[])
{
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(options_[1]);
}

}
