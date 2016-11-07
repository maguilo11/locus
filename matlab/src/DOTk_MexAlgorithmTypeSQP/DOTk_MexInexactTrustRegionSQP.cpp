/*
 * DOTk_MexInexactTrustRegionSQP.cpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <tr1/memory>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_InexactTrustRegionSQP.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"
#include "DOTk_MexParseAlgorithmTypeSQP.hpp"
#include "DOTk_MexInexactTrustRegionSQP.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"
#include "DOTk_InexactTrustRegionSqpSolverMng.hpp"

namespace dotk
{

DOTk_MexInexactTrustRegionSQP::DOTk_MexInexactTrustRegionSQP(const mxArray* options_[]) :
        dotk::DOTk_MexAlgorithmTypeSQP(options_[0]),
        m_ObjectiveFunctionOperators(NULL),
        m_EqualityConstraintOperators(NULL)
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
    double value = 0;
    dotk::mex::parseMaxTrustRegionRadius(options_, value);
    mng_.setMaxTrustRegionRadius(value);
    dotk::mex::parseMinTrustRegionRadius(options_, value);
    mng_.setMinTrustRegionRadius(value);
    dotk::mex::parseInitialTrustRegionRadius(options_, value);
    mng_.setTrustRegionRadius(value);
    dotk::mex::parseTrustRegionExpansionFactor(options_, value);
    mng_.setTrustRegionExpansionParameter(value);
    dotk::mex::parseTrustRegionContractionFactor(options_, value);
    mng_.setTrustRegionContractionParameter(value);
    dotk::mex::parseMinActualOverPredictedReductionRatio(options_, value);
    mng_.setMinActualOverPredictedReductionAllowed(value);
}

void DOTk_MexInexactTrustRegionSQP::setSqpKrylovSolversParameters(const mxArray* options_,
                                                                  dotk::DOTk_InexactTrustRegionSqpSolverMng & mng_)
{
    size_t value = 0;
    dotk::mex::parseSqpMaxNumDualProblemItr(options_, value);
    mng_.setMaxNumDualProblemItr(value);
    dotk::mex::parseSqpMaxNumTangentialProblemItr(options_, value);
    mng_.setMaxNumTangentialProblemItr(value);
    dotk::mex::parseSqpMaxNumQuasiNormalProblemItr(options_, value);
    mng_.setMaxNumQuasiNormalProblemItr(value);
    dotk::mex::parseSqpMaxNumTangentialSubProblemItr(options_, value);
    mng_.setMaxNumTangentialSubProblemItr(value);
}

void DOTk_MexInexactTrustRegionSQP::setAlgorithmParameters(const mxArray* options_,
                                                           dotk::DOTk_InexactTrustRegionSQP & algorithm_)
{
    double value = 0;
    dotk::mex::parseTangentialTolerance(options_, value);
    algorithm_.setTangentialTolerance(value);
    dotk::mex::parseDualProblemTolerance(options_, value);
    algorithm_.setDualProblemTolerance(value);
    dotk::mex::parseDualDotGradientTolerance(options_, value);
    algorithm_.setDualDotGradientTolerance(value);
    dotk::mex::parseToleranceContractionFactor(options_, value);
    algorithm_.setToleranceContractionFactor(value);
    dotk::mex::parsePredictedReductionParameter(options_, value);
    algorithm_.setPredictedReductionParameter(value);
    dotk::mex::parseMeritFunctionPenaltyParameter(options_, value);
    algorithm_.setMeritFunctionPenaltyParameter(value);
    dotk::mex::parseQuasiNormalProblemRelativeTolerance(options_, value);
    algorithm_.setQuasiNormalProblemRelativeTolerance(value);
    dotk::mex::parseTangentialToleranceContractionFactor(options_, value);
    algorithm_.setTangentialToleranceContractionFactor(value);
    dotk::mex::parseActualOverPredictedReductionTolerance(options_, value);
    algorithm_.setActualOverPredictedReductionTolerance(value);
    dotk::mex::parseMaxEffectiveTangentialOverTrialStepRatio(options_, value);
    algorithm_.setMaxEffectiveTangentialOverTrialStepRatio(value);
    dotk::mex::parseTangentialSubProbLeftPrecProjectionTolerance(options_, value);
    algorithm_.setTangentialSubProbLeftPrecProjectionTolerance(value);
    dotk::mex::parseQuasiNormalProblemTrustRegionRadiusPenaltyParameter(options_, value);
    algorithm_.setQuasiNormalProblemTrustRegionRadiusPenaltyParameter(value);
}

void DOTk_MexInexactTrustRegionSQP::solveTypeEqualityConstrainedLinearProgramming(const mxArray* input_[],
                                                                                  mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeSQP::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraintOperators);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    this->setTrustRegionParameters(input_[0], *mng);

    std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));

    this->setSqpKrylovSolversParameters(input_[0], *solver_mng);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    hessian->setFullSpaceHessian();
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP algorithm(hessian, mng, solver_mng);
    this->setAlgorithmParameters(input_[0], algorithm);
    algorithm.getMin();

    this->gatherOutputDataTypeLP(algorithm, *mng, output_);
}

void DOTk_MexInexactTrustRegionSQP::solveTypeEqualityConstrainedNonlinearProgramming(const mxArray* input_[],
                                                                                     mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeSQP::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraintOperators);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildStateContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    this->setTrustRegionParameters(input_[0], *mng);

    std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    this->setSqpKrylovSolversParameters(input_[0], *solver_mng);

    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();

    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP algorithm(hessian, mng, solver_mng);
    this->setAlgorithmParameters(input_[0], algorithm);

    algorithm.getMin();

    this->gatherOutputDataTypeNLP(algorithm, *mng, output_);
}

void DOTk_MexInexactTrustRegionSQP::gatherOutputDataTypeLP(dotk::DOTk_InexactTrustRegionSQP & algorithm_,
                                                           dotk::DOTk_TrustRegionMngTypeELP & mng_,
                                                           mxArray* output_[])
{
    // Create memory allocation for output struct
    const char *field_names[8] =
        { "Iterations", "ObjectiveFunctionValue", "Primal", "Dual", "Gradient", "NormGradien", "TrialStep",
                "NormTrialStep" };
    output_[0] = mxCreateStructMatrix(1, 1, 8, field_names);

    dotk::DOTk_MexArrayPtr itr(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(itr.get()))[0] = algorithm_.getNumItrDone();
    mxSetField(output_[0], 0, "Iterations", itr.get());
    itr.release();

    dotk::DOTk_MexArrayPtr objective_function_value(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(objective_function_value.get())[0] = mng_.getNewObjectiveFunctionValue();
    mxSetField(output_[0], 0, "ObjectiveFunction", objective_function_value.get());
    objective_function_value.release();

    size_t num_primal = mng_.getNewPrimal()->size();
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(num_primal, 1, mxREAL));
    mng_.getNewPrimal()->gather(mxGetPr(primal.get()));
    mxSetField(output_[0], 0, "Primal", primal.get());
    primal.release();

    size_t num_dual = mng_.getNewDual()->size();
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(num_dual, 1, mxREAL));
    mng_.getNewDual()->gather(mxGetPr(dual.get()));
    mxSetField(output_[0], 0, "Dual", dual.get());
    dual.release();

    dotk::DOTk_MexArrayPtr gradient(mxCreateDoubleMatrix(num_primal, 1, mxREAL));
    mng_.getNewGradient()->gather(mxGetPr(gradient.get()));
    mxSetField(output_[0], 0, "Gradient", gradient.get());
    gradient.release();

    dotk::DOTk_MexArrayPtr norm_gradient(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_gradient.get())[0] = mng_.getNewGradient()->norm();
    mxSetField(output_[0], 0, "NormGradient", norm_gradient.get());
    norm_gradient.release();

    dotk::DOTk_MexArrayPtr trial_step(mxCreateDoubleMatrix(num_primal, 1, mxREAL));
    mng_.getTrialStep()->gather(mxGetPr(trial_step.get()));
    mxSetField(output_[0], 0, "TrialStep", trial_step.get());
    trial_step.release();

    dotk::DOTk_MexArrayPtr norm_trial_step(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_trial_step.get())[0] = mng_.getTrialStep()->norm();
    mxSetField(output_[0], 0, "NormTrialStep", norm_trial_step.get());
    norm_trial_step.release();
}

void DOTk_MexInexactTrustRegionSQP::gatherOutputDataTypeNLP(dotk::DOTk_InexactTrustRegionSQP & algorithm_,
                                                            dotk::DOTk_TrustRegionMngTypeELP & mng_,
                                                            mxArray* output_[])
{
    // Create memory allocation for output struct
    const char *field_names[13] =
        { "Iterations", "ObjectiveFunctionValue", "State", "Control", "Dual", "StateGradient", "NormStateGradient",
                "ControlGradient", "NormControlGradient", "StateTrialStep", "NormStateTrialStep", "ControlTrialStep",
                "NormControlTrialStep" };
    output_[0] = mxCreateStructMatrix(1, 1, 13, field_names);

    dotk::DOTk_MexArrayPtr itr(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(itr.get()))[0] = algorithm_.getNumItrDone();
    mxSetField(output_[0], 0, "Iterations", itr.get());
    itr.release();

    dotk::DOTk_MexArrayPtr objective_function_value(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(objective_function_value.get())[0] = mng_.getNewObjectiveFunctionValue();
    mxSetField(output_[0], 0, "ObjectiveFunction", objective_function_value.get());
    objective_function_value.release();

    size_t num_states = mng_.getNewPrimal()->state()->size();
    dotk::DOTk_MexArrayPtr state(mxCreateDoubleMatrix(num_states, 1, mxREAL));
    mng_.getNewPrimal()->state()->gather(mxGetPr(state.get()));
    mxSetField(output_[0], 0, "State", state.get());
    state.release();

    size_t num_control = mng_.getNewPrimal()->control()->size();
    dotk::DOTk_MexArrayPtr control(mxCreateDoubleMatrix(num_control, 1, mxREAL));
    mng_.getNewPrimal()->control()->gather(mxGetPr(control.get()));
    mxSetField(output_[0], 0, "Control", control.get());
    control.release();

    size_t num_dual = mng_.getNewDual()->size();
    dotk::DOTk_MexArrayPtr dual(mxCreateDoubleMatrix(num_dual, 1, mxREAL));
    mng_.getNewDual()->gather(mxGetPr(dual.get()));
    mxSetField(output_[0], 0, "Dual", dual.get());
    dual.release();

    dotk::DOTk_MexArrayPtr state_gradient(mxCreateDoubleMatrix(num_states, 1, mxREAL));
    mng_.getNewGradient()->state()->gather(mxGetPr(state_gradient.get()));
    mxSetField(output_[0], 0, "StateGradient", state_gradient.get());
    state_gradient.release();

    dotk::DOTk_MexArrayPtr norm_state_gradient(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_state_gradient.get())[0] = mng_.getNewGradient()->state()->norm();
    mxSetField(output_[0], 0, "NormStateGradient", norm_state_gradient.get());
    norm_state_gradient.release();

    dotk::DOTk_MexArrayPtr control_gradient(mxCreateDoubleMatrix(num_control, 1, mxREAL));
    mng_.getNewGradient()->control()->gather(mxGetPr(control_gradient.get()));
    mxSetField(output_[0], 0, "ControlGradient", control_gradient.get());
    control_gradient.release();

    dotk::DOTk_MexArrayPtr norm_control_gradient(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_control_gradient.get())[0] = mng_.getNewGradient()->control()->norm();
    mxSetField(output_[0], 0, "NormControlGradient", norm_control_gradient.get());
    norm_control_gradient.release();

    dotk::DOTk_MexArrayPtr state_trial_step(mxCreateDoubleMatrix(num_states, 1, mxREAL));
    mng_.getTrialStep()->state()->gather(mxGetPr(state_trial_step.get()));
    mxSetField(output_[0], 0, "StateTrialStep", state_trial_step.get());
    state_trial_step.release();

    dotk::DOTk_MexArrayPtr norm_state_trial_step(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_state_trial_step.get())[0] = mng_.getTrialStep()->state()->norm();
    mxSetField(output_[0], 0, "NormStateTrialStep", norm_state_trial_step.get());
    norm_state_trial_step.release();

    dotk::DOTk_MexArrayPtr control_trial_step(mxCreateDoubleMatrix(num_control, 1, mxREAL));
    mng_.getTrialStep()->control()->gather(mxGetPr(control_trial_step.get()));
    mxSetField(output_[0], 0, "ControlTrialStep", control_trial_step.get());
    control_trial_step.release();

    dotk::DOTk_MexArrayPtr norm_control_trial_step(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_control_trial_step.get())[0] = mng_.getTrialStep()->control()->norm();
    mxSetField(output_[0], 0, "NormControlTrialStep", norm_control_trial_step.get());
    norm_control_trial_step.release();
}

void DOTk_MexInexactTrustRegionSQP::clear()
{
    m_ObjectiveFunctionOperators.release();
    m_EqualityConstraintOperators.release();
}

void DOTk_MexInexactTrustRegionSQP::initialize(const mxArray* options_[])
{
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunctionOperators);
    dotk::mex::parseEqualityConstraint(options_[1], m_EqualityConstraintOperators);
}

}
