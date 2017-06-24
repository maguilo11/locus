/*
 * DOTk_MexInexactNewtonTypeLS.cpp
 *
 *  Created on: Apr 25, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexInexactNewtonTypeLS.hpp"
#include "DOTk_MexNumDiffHessianFactory.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_LineSearchInexactNewton.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace dotk
{

DOTk_MexInexactNewtonTypeLS::DOTk_MexInexactNewtonTypeLS(const mxArray* options_[]) :
        dotk::DOTk_MexAlgorithmTypeNewton(options_[0]),
        m_MaxNumLineSearchItr(10),
        m_LineSearchContractionFactor(0.5),
        m_LineSearchStagnationTolerance(1e-8),
        m_LineSearchMethod(dotk::types::LINE_SEARCH_DISABLED),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr)
        {
            this->initialize(options_);
        }

DOTk_MexInexactNewtonTypeLS::~DOTk_MexInexactNewtonTypeLS()
{
    this->clear();
}

void DOTk_MexInexactNewtonTypeLS::solve(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeNewton::getProblemType();

    switch(type)
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
        case dotk::types::TYPE_NLP_BOUND:
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
            std::string msg(" DOTk/MEX ERROR: Invalid Problem Type for Line Search Based Newton Method. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

size_t DOTk_MexInexactNewtonTypeLS::getMaxNumLineSearchItr() const
{
    return (m_MaxNumLineSearchItr);
}

double DOTk_MexInexactNewtonTypeLS::getLineSearchContractionFactor() const
{
    return (m_LineSearchContractionFactor);
}

double DOTk_MexInexactNewtonTypeLS::getLineSearchStagnationTolerance() const
{
    return (m_LineSearchStagnationTolerance);
}

dotk::types::line_search_t DOTk_MexInexactNewtonTypeLS::getLineSearchMethod() const
{
    return (m_LineSearchMethod);
}

void DOTk_MexInexactNewtonTypeLS::setAlgorithmParameters(dotk::DOTk_LineSearchInexactNewton & algorithm_)
{
    size_t max_num_itr = DOTk_MexAlgorithmTypeNewton::getMaxNumAlgorithmItr();
    algorithm_.setMaxNumItr(max_num_itr);
    Real objective_tolerance = DOTk_MexAlgorithmTypeNewton::getObjectiveFunctionTolerance();
    algorithm_.setObjectiveFuncTol(objective_tolerance);
    Real gradient_tolerance = DOTk_MexAlgorithmTypeNewton::getGradientTolerance();
    algorithm_.setGradientTol(gradient_tolerance);
    Real trial_step_tolerance = DOTk_MexAlgorithmTypeNewton::getTrialStepTolerance();
    algorithm_.setTrialStepTol(trial_step_tolerance);
    Real krylov_solver_relative_tolerance = DOTk_MexAlgorithmTypeNewton::getKrylovSolverRelativeTolerance();
    algorithm_.setRelativeTolerance(krylov_solver_relative_tolerance);
}

void DOTk_MexInexactNewtonTypeLS::setLineSearchMethodParameters(const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_)
{
    size_t max_num_itr = this->getMaxNumLineSearchItr();
    step_->setMaxNumIterations(max_num_itr);
    Real contraction_factor = this->getLineSearchContractionFactor();
    step_->setContractionFactor(contraction_factor);
    Real stagnation_tolerance = this->getLineSearchStagnationTolerance();
    step_->setStagnationTolerance(stagnation_tolerance);
}

void DOTk_MexInexactNewtonTypeLS::solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: control vector
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateUserDefinedControl(controls);

    // Set line search step manager
    dotk::types::line_search_t line_search_type = this->getLineSearchMethod();
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    step->build(primal, line_search_type);
    this->setLineSearchMethodParameters(step);

    // Set objective function operators
    dotk::types::problem_t problem_type = dotk::DOTk_MexAlgorithmTypeNewton::getProblemType();
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction> objective =
            std::make_shared<dotk::DOTk_MexObjectiveFunction>(m_ObjectiveFunction, problem_type);

    // Set line search algorithm data manager, gradient, and Hessian
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng =
            std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);
    dotk::mex::buildGradient(input_[0], mng);
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian =
            std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective);
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    // Initialize line search algorithm
    dotk::DOTk_LineSearchInexactNewton algorithm(hessian, step, mng);
    dotk::mex::buildKrylovSolver(input_[0], primal, algorithm);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    algorithm.getMin();

    DOTk_MexAlgorithmTypeNewton::gatherOutputData(algorithm, *mng, output_);
}

void DOTk_MexInexactNewtonTypeLS::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    // Set objective function operators
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    dotk::types::line_search_t line_search_type = this->getLineSearchMethod();
    step->build(primal, line_search_type);
    this->setLineSearchMethodParameters(step);

    // Set objective funciton and equality constraint operators
    dotk::types::problem_t problem_type = DOTk_MexAlgorithmTypeNewton::getProblemType();
    std::shared_ptr<dotk::DOTk_MexObjectiveFunction> objective =
            std::make_shared<dotk::DOTk_MexObjectiveFunction>(m_ObjectiveFunction, problem_type);
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::shared_ptr<dotk::DOTk_MexEqualityConstraint> equality =
            std::make_shared<dotk::DOTk_MexEqualityConstraint>(m_EqualityConstraint, problem_type);

    // Set line search algorithm data manager
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeUNP> mng =
            std::make_shared<dotk::DOTk_LineSearchMngTypeUNP>(primal, objective, equality);
    dotk::mex::buildGradient(input_[0], mng);
    std::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian =
            std::make_shared<dotk::NumericallyDifferentiatedHessian>(primal, objective, equality);
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    // Initialize line search algorithm
    dotk::DOTk_LineSearchInexactNewton algorithm(hessian, step, mng);
    dotk::mex::buildKrylovSolver(input_[0], primal, algorithm);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    algorithm.getMin();

    DOTk_MexAlgorithmTypeNewton::gatherOutputData(algorithm, *mng, output_);
}

void DOTk_MexInexactNewtonTypeLS::clear()
{
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
}

void DOTk_MexInexactNewtonTypeLS::initialize(const mxArray* options_[])
{
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);

    m_LineSearchMethod = dotk::mex::parseLineSearchMethod(options_[0]);
    m_MaxNumLineSearchItr = dotk::mex::parseMaxNumLineSearchItr(options_[0]);
    m_LineSearchContractionFactor = dotk::mex::parseLineSearchContractionFactor(options_[0]);
    m_LineSearchStagnationTolerance = dotk::mex::parseLineSearchStagnationTolerance(options_[0]);

}

}
