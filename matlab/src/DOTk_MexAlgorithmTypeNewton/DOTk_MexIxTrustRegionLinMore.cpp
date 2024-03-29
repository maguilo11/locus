/*
 * DOTk_MexIxTrustRegionLinMore.cpp
 *
 *  Created on: Sep 7, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_SteihaugTointStepMng.hpp"
#include "DOTk_SteihaugTointDataMng.hpp"
#include "DOTk_SteihaugTointLinMore.hpp"
#include "DOTk_SteihaugTointProjGradStep.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexKrylovSolverParser.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexIxTrustRegionLinMore.hpp"
#include "DOTk_MexNumDiffHessianFactory.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

namespace dotk
{


DOTk_MexIxTrustRegionLinMore::DOTk_MexIxTrustRegionLinMore(const mxArray* options_[]) :
        dotk::DOTk_MexSteihaugTointNewton(options_),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED),
        m_MaxNumSteihaugTointSolverItr(200),
        m_SolverRelativeTolerance(0.1),
        m_SolverRelativeToleranceExponential(0.5),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr)
{
    this->initializeInexactTrustRegionLinMore(options_);
}

DOTk_MexIxTrustRegionLinMore::~DOTk_MexIxTrustRegionLinMore()
{
    this->clear();
}

void DOTk_MexIxTrustRegionLinMore::clear()
{
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
}

void DOTk_MexIxTrustRegionLinMore::initializeInexactTrustRegionLinMore(const mxArray* options_[])
{
    m_ProblemType = dotk::mex::parseProblemType(options_[0]);
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);
    m_MaxNumSteihaugTointSolverItr = dotk::mex::parseMaxNumKrylovSolverItr(options_[0]);
    m_SolverRelativeTolerance = dotk::mex::parseKrylovSolverRelativeTolerance(options_[0]);
    m_SolverRelativeToleranceExponential = dotk::mex::parseRelativeToleranceExponential(options_[0]);
}

size_t DOTk_MexIxTrustRegionLinMore::getMaxNumSteihaugTointSolverItr() const
{
    return (m_MaxNumSteihaugTointSolverItr);
}

double DOTk_MexIxTrustRegionLinMore::getSolverRelativeTolerance() const
{
    return (m_SolverRelativeTolerance);
}

double DOTk_MexIxTrustRegionLinMore::getSolverRelativeToleranceExponential() const
{
    return (m_SolverRelativeToleranceExponential);
}

void DOTk_MexIxTrustRegionLinMore::solve(const mxArray* input_[], mxArray* output_[])
{
    switch(m_ProblemType)
    {
        case dotk::types::TYPE_ULP:
        {
            this->solveTypeLinearProgramming(input_, output_);
            break;
        }
        case dotk::types::TYPE_UNLP:
        {
            this->solveTypeNonlinearProgramming(input_, output_);
            break;
        }
        case dotk::types::TYPE_LP_BOUND:
        {
            this->solveTypeBoundLinearProgramming(input_, output_);
            break;
        }
        case dotk::types::TYPE_NLP_BOUND:
        {
            this->solveTypeBoundNonlinearProgramming(input_, output_);
            break;
        }
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_CNLP:
        case dotk::types::TYPE_ILP:
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::string msg(" DOTk/MEX ERROR: Invalid Problem Type for Inexact Lin-More Trust Region Newton Algorithm. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexIxTrustRegionLinMore::solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: control vector
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateUserDefinedControl(controls);

    // Set objective function and data manager
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective = std::make_shared<dotk::DOTk_MexObjectiveFunction>(m_ObjectiveFunction, m_ProblemType);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data = std::make_shared<dotk::DOTk_SteihaugTointDataMng>(primal, objective);
    dotk::mex::buildGradient(input_[0], data);

    // Set numerically differentiated Hessian
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    // Set trust region step manager
    std::shared_ptr<dotk::DOTk_SteihaugTointStepMng>
        step = std::make_shared<dotk::DOTk_SteihaugTointStepMng>(primal, hessian);
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    // Initialize trust region algorithm
    dotk::DOTk_SteihaugTointLinMore algorithm(data, step);
    this->setIxLinMoreAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionLinMore::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    // Set objective function and equality constraint
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective = std::make_shared<dotk::DOTk_MexObjectiveFunction>(m_ObjectiveFunction, m_ProblemType);
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality = std::make_shared<dotk::DOTk_MexEqualityConstraint>(m_EqualityConstraint, m_ProblemType);

    // Set data manager
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data = std::make_shared<dotk::DOTk_SteihaugTointDataMng>(primal, objective, equality);
    dotk::mex::buildGradient(input_[0], data);

    // Set numerically differentiated Hessian
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective, equality);
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    std::shared_ptr<dotk::DOTk_SteihaugTointStepMng>
        step = std::make_shared<dotk::DOTk_SteihaugTointStepMng>(primal, hessian);
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    dotk::DOTk_SteihaugTointLinMore algorithm(data, step);
    this->setIxLinMoreAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionLinMore::solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    // Set objective function and data manager
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective = std::make_shared<dotk::DOTk_MexObjectiveFunction>(m_ObjectiveFunction, m_ProblemType);
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data = std::make_shared<dotk::DOTk_SteihaugTointDataMng>(primal, objective);
    dotk::mex::buildGradient(input_[0], data);

    // Set numerically differentiated Hessian
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    // Set trust region step manager
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step = std::make_shared<dotk::DOTk_SteihaugTointProjGradStep>(primal, hessian);
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    // Initialize trust region algorithm
    dotk::DOTk_SteihaugTointLinMore algorithm(data, step);
    this->setIxLinMoreAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionLinMore::solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    // Set objective function and equality constraint
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective = std::make_shared<dotk::DOTk_MexObjectiveFunction>(m_ObjectiveFunction, m_ProblemType);
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality = std::make_shared<dotk::DOTk_MexEqualityConstraint>(m_EqualityConstraint, m_ProblemType);

    // Set traut region data manager
    std::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data = std::make_shared<dotk::DOTk_SteihaugTointDataMng>(primal, objective, equality);
    dotk::mex::buildGradient(input_[0], data);

    std::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian = std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective, equality);
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    // Set trust region step manager
    std::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step = std::make_shared<dotk::DOTk_SteihaugTointProjGradStep>(primal, hessian);
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    // Initialize trust region algorithm
    dotk::DOTk_SteihaugTointLinMore algorithm(data, step);
    this->setIxLinMoreAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionLinMore::setIxLinMoreAlgorithmParameters(dotk::DOTk_SteihaugTointLinMore & algorithm_)
{
    dotk::DOTk_MexSteihaugTointNewton::setAlgorithmParameters(algorithm_);

    double relative_tolerance = this->getSolverRelativeTolerance();
    algorithm_.setSolverRelativeTolerance(relative_tolerance);
    size_t max_num_steihaug_toint_itr = this->getMaxNumSteihaugTointSolverItr();
    algorithm_.setSolverMaxNumItr(max_num_steihaug_toint_itr);
    double relative_tolerance_exponential = this->getSolverRelativeToleranceExponential();
    algorithm_.setSolverRelativeToleranceExponential(relative_tolerance_exponential);

    algorithm_.printDiagnosticsEveryItrAndSolutionAtTheEnd();
}

}
