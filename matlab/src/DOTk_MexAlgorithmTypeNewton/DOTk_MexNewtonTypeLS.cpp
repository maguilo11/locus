/*
 * DOTk_MexNewtonTypeLS.cpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexVector.hpp"
#include "DOTk_MexNewtonTypeLS.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexHessianFactory.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_LineSearchInexactNewton.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_MexNewtonTypeLS::DOTk_MexNewtonTypeLS(const mxArray* options_[]) :
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

DOTk_MexNewtonTypeLS::~DOTk_MexNewtonTypeLS()
{
    this->clear();
}

void DOTk_MexNewtonTypeLS::solve(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeNewton::getProblemType();

    switch (type)
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

size_t DOTk_MexNewtonTypeLS::getMaxNumLineSearchItr() const
{
    return (m_MaxNumLineSearchItr);
}

double DOTk_MexNewtonTypeLS::getLineSearchContractionFactor() const
{
    return (m_LineSearchContractionFactor);
}

double DOTk_MexNewtonTypeLS::getLineSearchStagnationTolerance() const
{
    return (m_LineSearchStagnationTolerance);
}

dotk::types::line_search_t DOTk_MexNewtonTypeLS::getLineSearchMethod() const
{
    return (m_LineSearchMethod);
}

void DOTk_MexNewtonTypeLS::setAlgorithmParameters(dotk::DOTk_LineSearchInexactNewton & algorithm_)
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

void DOTk_MexNewtonTypeLS::setLineSearchMethodParameters
(const std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_)
{
    size_t max_num_itr = this->getMaxNumLineSearchItr();
    step_->setMaxNumIterations(max_num_itr);
    Real contraction_factor = this->getLineSearchContractionFactor();
    step_->setContractionFactor(contraction_factor);
    Real stagnation_tolerance = this->getLineSearchStagnationTolerance();
    step_->setStagnationTolerance(stagnation_tolerance);
}

void DOTk_MexNewtonTypeLS::solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: control vector
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateUserDefinedControl(controls);

    // Set objective function and data manager
    dotk::types::problem_t problem_type = DOTk_MexAlgorithmTypeNewton::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, problem_type));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
        data_mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    // Set gradient and Hessian computation method
    dotk::mex::buildGradient(input_[0], data_mng);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::mex::buildHessian(input_[0], hessian);

    // Set line search step manager
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    dotk::types::line_search_t line_search_type = this->getLineSearchMethod();
    step->build(primal, line_search_type);
    this->setLineSearchMethodParameters(step);

    // Initialize line search algorithm
    dotk::DOTk_LineSearchInexactNewton algorithm(hessian, step, data_mng);
    dotk::mex::buildKrylovSolver(input_[0], primal, algorithm);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    algorithm.getMin();

    DOTk_MexAlgorithmTypeNewton::gatherOutputData(algorithm, *data_mng, output_);
}

void DOTk_MexNewtonTypeLS::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: state and control vectors
    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector states(num_states, 0.);
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);

    // Allocate DOTk data structures
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateUserDefinedState(states);
    primal->allocateUserDefinedControl(controls);

    // Set objective function and equality constraint
    dotk::types::problem_t problem_type = DOTk_MexAlgorithmTypeNewton::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, problem_type));
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality(new dotk::DOTk_MexEqualityConstraint(m_EqualityConstraint, problem_type));

    // Set data manager and gradient/Hessian computation methods
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeUNP>
        data_mng(new dotk::DOTk_LineSearchMngTypeUNP(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], data_mng);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::mex::buildHessian(input_[0], hessian);

    // Set line search step manager
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    dotk::types::line_search_t line_search_type = this->getLineSearchMethod();
    step->build(primal, line_search_type);
    this->setLineSearchMethodParameters(step);

    // Initialize line search algorithm
    dotk::DOTk_LineSearchInexactNewton algorithm(hessian, step, data_mng);
    dotk::mex::buildKrylovSolver(input_[0], primal, algorithm);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    algorithm.getMin();

    DOTk_MexAlgorithmTypeNewton::gatherOutputData(algorithm, *data_mng, output_);
}

void DOTk_MexNewtonTypeLS::clear()
{
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
}

void DOTk_MexNewtonTypeLS::initialize(const mxArray* options_[])
{
    m_LineSearchMethod = dotk::mex::parseLineSearchMethod(options_[0]);
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);
    m_MaxNumLineSearchItr = dotk::mex::parseMaxNumLineSearchItr(options_[0]);
    m_LineSearchContractionFactor = dotk::mex::parseLineSearchContractionFactor(options_[0]);
    m_LineSearchStagnationTolerance = dotk::mex::parseLineSearchStagnationTolerance(options_[0]);
}

}
