/*
 * DOTk_MexGradientProjection.cpp
 *
 *  Created on: Oct 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexVector.hpp"
#include "DOTk_MexNonlinearCG.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexGradientProjection.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_GradientProjectionMethod.hpp"

namespace dotk
{

DOTk_MexGradientProjection::DOTk_MexGradientProjection(const mxArray* options_[]) :
        m_MaxNumIterations(5000),
        m_MaxNumLineSearchIterations(10),
        m_ObjectiveTolerance(1e-8),
        m_ProjectedGradientTolerance(1e-8),
        m_LineSearchContractionFactor(0.5),
        m_LineSearchStagnationTolerance(1e-8),
        m_ProblemType(dotk::types::problem_t::PROBLEM_TYPE_UNDEFINED),
        m_LineSearchMethod(dotk::types::LINE_SEARCH_DISABLED),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr)
{
    this->initialize(options_);
}

DOTk_MexGradientProjection::~DOTk_MexGradientProjection()
{
    this->clear();
}

void DOTk_MexGradientProjection::clear()
{
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
}

void DOTk_MexGradientProjection::initialize(const mxArray* options_[])
{
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);

    m_ProblemType = dotk::mex::parseProblemType(options_[0]);
    m_LineSearchMethod = dotk::mex::parseLineSearchMethod(options_[0]);
    m_MaxNumIterations = dotk::mex::parseMaxNumOuterIterations(options_[0]);
    m_ObjectiveTolerance = dotk::mex::parseObjectiveTolerance(options_[0]);
    m_ProjectedGradientTolerance = dotk::mex::parseGradientTolerance(options_[0]);

    m_MaxNumLineSearchIterations = dotk::mex::parseMaxNumLineSearchItr(options_[0]);
    m_LineSearchContractionFactor = dotk::mex::parseLineSearchContractionFactor(options_[0]);
    m_LineSearchStagnationTolerance = dotk::mex::parseLineSearchStagnationTolerance(options_[0]);
}

void DOTk_MexGradientProjection::solve(const mxArray* input_[], mxArray* output_[])
{
    switch(m_ProblemType)
    {
        case dotk::types::TYPE_ULP:
        {
            this->solveLinearProgrammingProblem(input_, output_);
            break;
        }
        case dotk::types::TYPE_UNLP:
        {
            this->solveNonlinearProgrammingProblem(input_, output_);
            break;
        }
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_CNLP:
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_NLP_BOUND:
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::TYPE_ILP:
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::string msg("\n\n DOTk/MEX ERROR: Invalid Problem Type for Gradient Projection Method. See Users' Manual. \n\n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexGradientProjection::solveLinearProgrammingProblem(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: control vectors
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);

    // Allocate DOTk data structures
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateUserDefinedControl(controls);

    // Set lower bounds on control variables
    mxArray* mx_lower_bound = dotk::mex::parseControlLowerBound(input_[0]);
    dotk::MexVector lower_bound(mx_lower_bound);
    mxDestroyArray(mx_lower_bound);
    primal->setControlLowerBound(lower_bound);

    // Set upper bounds on control variables
    mxArray* mx_upper_bound = dotk::mex::parseControlUpperBound(input_[0]);
    dotk::MexVector upper_bound(mx_upper_bound);
    mxDestroyArray(mx_upper_bound);
    primal->setControlUpperBound(upper_bound);

    // Set line search step manager
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    step->build(primal, m_LineSearchMethod);
    step->setMaxNumIterations(m_MaxNumLineSearchIterations);
    step->setContractionFactor(m_LineSearchContractionFactor);
    step->setStagnationTolerance(m_LineSearchStagnationTolerance);

    // Set line search algorithm data manager
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective = std::make_shared<dotk::DOTk_MexObjectiveFunction>(m_ObjectiveFunction, m_ProblemType);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
        mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    dotk::mex::buildGradient(input_[0], mng);

    // Initialize algorithm
    dotk::GradientProjectionMethod algorithm(primal, step, mng);
    algorithm.setMaxNumIterations(m_MaxNumIterations);
    algorithm.setObjectiveTolerance(m_ObjectiveTolerance);
    algorithm.setProjectedGradientTolerance(m_ProjectedGradientTolerance);
    algorithm.printDiagnostics();

    algorithm.getMin();

    this->outputData(algorithm, *mng, output_);
}

void DOTk_MexGradientProjection::solveNonlinearProgrammingProblem(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: state and control vectors
    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector states(num_states, 0.);
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);

    // Allocate DOTk data structures
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateUserDefinedState(states);
    primal->allocateUserDefinedControl(controls);

    // Set lower bounds on control variables
    mxArray* mx_lower_bound = dotk::mex::parseControlLowerBound(input_[0]);
    dotk::MexVector lower_bound(mx_lower_bound);
    mxDestroyArray(mx_lower_bound);
    primal->setControlLowerBound(lower_bound);

    // Set upper bounds on control variables
    mxArray* mx_upper_bound = dotk::mex::parseControlUpperBound(input_[0]);
    dotk::MexVector upper_bound(mx_upper_bound);
    mxDestroyArray(mx_upper_bound);
    primal->setControlUpperBound(upper_bound);

    // Set line search step manager
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    step->build(primal, m_LineSearchMethod);
    step->setMaxNumIterations(m_MaxNumLineSearchIterations);
    step->setContractionFactor(m_LineSearchContractionFactor);
    step->setStagnationTolerance(m_LineSearchStagnationTolerance);

    // Set objective function, equality constraint, and line search algorithm data manager
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective = std::make_shared<dotk::DOTk_MexObjectiveFunction>(m_ObjectiveFunction, m_ProblemType);
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality = std::make_shared<dotk::DOTk_MexEqualityConstraint>(m_EqualityConstraint, m_ProblemType);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeUNP>
        mng = std::make_shared<dotk::DOTk_LineSearchMngTypeUNP>(primal, objective, equality);
    dotk::mex::buildGradient(input_[0], mng);

    // Initialize gradient projection algorithm
    dotk::GradientProjectionMethod algorithm(primal, step, mng);
    algorithm.setMaxNumIterations(m_MaxNumIterations);
    algorithm.setObjectiveTolerance(m_ObjectiveTolerance);
    algorithm.setProjectedGradientTolerance(m_ProjectedGradientTolerance);
    algorithm.printDiagnostics();

    algorithm.getMin();

    this->outputData(algorithm, *mng, output_);
}

void DOTk_MexGradientProjection::outputData(const dotk::GradientProjectionMethod & algorithm_,
                                            const dotk::DOTk_LineSearchAlgorithmsDataMng & mng_,
                                            mxArray* outputs_[])
{
    // Create memory allocation for output struct
    const char *field_names[7] =
        { "Iterations", "ObjectiveFunctionValue", "Control", "Gradient", "NormGradient", "ProjectedGradient", "NormProjectedGradient" };
    outputs_[0] = mxCreateStructMatrix(1, 1, 7, field_names);

    mxArray* number_iterations = mxCreateDoubleScalar(algorithm_.getIterationCount());
    mxSetField(outputs_[0], 0, "Iterations", number_iterations);
    mxDestroyArray(number_iterations);

    mxArray* objective_function_value = mxCreateDoubleScalar(mng_.getNewObjectiveFunctionValue());
    mxSetField(outputs_[0], 0, "ObjectiveFunctionValue", objective_function_value);
    mxDestroyArray(objective_function_value);

    dotk::MexVector & control = dynamic_cast<dotk::MexVector &>(*mng_.getNewPrimal());
    mxSetField(outputs_[0], 0, "Control", control.array());

    dotk::MexVector & gradient = dynamic_cast<dotk::MexVector &>(*mng_.getNewGradient());
    mxSetField(outputs_[0], 0, "Gradient", gradient.array());

    mxArray* norm_gradient = mxCreateDoubleScalar(gradient.norm());
    mxSetField(outputs_[0], 0, "NormGradient", norm_gradient);
    mxDestroyArray(norm_gradient);

    dotk::MexVector & projected_gradient = dynamic_cast<dotk::MexVector &>(*mng_.getTrialStep());
    mxSetField(outputs_[0], 0, "ProjectedGradient", projected_gradient.array());

    mxArray* norm_projected_gradient = mxCreateDoubleScalar(projected_gradient.norm());
    mxSetField(outputs_[0], 0, "NormProjectedGradient", norm_projected_gradient);
    mxDestroyArray(norm_projected_gradient);
}

}
