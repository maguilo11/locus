/*
 * DOTk_MexIxTrustRegionKelleySachs.cpp
 *
 *  Created on: Apr 17, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <limits>


#include "DOTk_Primal.hpp"
#include "DOTk_KelleySachsStepMng.hpp"
#include "DOTk_SteihaugTointDataMng.hpp"
#include "DOTk_SteihaugTointKelleySachs.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexKrylovSolverParser.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexNumDiffHessianFactory.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"
#include "DOTk_MexIxTrustRegionKelleySachs.hpp"

namespace dotk
{

DOTk_MexIxTrustRegionKelleySachs::DOTk_MexIxTrustRegionKelleySachs(const mxArray* options_[]) :
        dotk::DOTk_MexSteihaugTointNewton(options_),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED),
        m_MaxNumUpdates(10),
        m_MaxNumSteihaugTointSolverItr(200),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr)
{
    this->initializeIxKelleySachsTrustRegion(options_);
}

DOTk_MexIxTrustRegionKelleySachs::~DOTk_MexIxTrustRegionKelleySachs()
{
    this->clear();
}

void DOTk_MexIxTrustRegionKelleySachs::clear()
{
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
}

void DOTk_MexIxTrustRegionKelleySachs::initializeIxKelleySachsTrustRegion(const mxArray* options_[])
{
    m_ProblemType = dotk::mex::parseProblemType(options_[0]);
    m_MaxNumUpdates = dotk::mex::parseMaxNumUpdates(options_[0]);
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);
    m_MaxNumSteihaugTointSolverItr = dotk::mex::parseMaxNumKrylovSolverItr(options_[0]);
}

size_t DOTk_MexIxTrustRegionKelleySachs::getMaxNumUpdates() const
{
    return (m_MaxNumUpdates);
}

size_t DOTk_MexIxTrustRegionKelleySachs::getMaxNumSteihaugTointSolverItr() const
{
    return (m_MaxNumSteihaugTointSolverItr);
}

void DOTk_MexIxTrustRegionKelleySachs::solve(const mxArray* input_[], mxArray* output_[])
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
            std::string msg(" DOTk/MEX ERROR: Invalid Problem Type for Kelley-Sachs Trust Region Newton Algorithm. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexIxTrustRegionKelleySachs::solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: control vector
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateUserDefinedControl(controls);

    // Disable lower and upper bounds
    const double DISABLED_LOWER_BOUND = -std::numeric_limits<double>::max();
    primal->setControlLowerBound(DISABLED_LOWER_BOUND);
    const double DISABLED_UPPER_BOUND = std::numeric_limits<double>::max();
    primal->setControlUpperBound(DISABLED_UPPER_BOUND);

    // Set objective function and trust region data manager
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, m_ProblemType));
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
    data(new dotk::DOTk_SteihaugTointDataMng(primal, objective));
    dotk::mex::buildGradient(input_[0], data);

    // Set numerically differentiated Hessian
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    // Set trust region step manager
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step(new dotk::DOTk_KelleySachsStepMng(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    // Initialize trust region algorithm
    dotk::DOTk_SteihaugTointKelleySachs algorithm(data, step);
    this->setIxKelleySachsAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionKelleySachs::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    // Disable lower and upper bounds
    const double DISABLED_LOWER_BOUND = -std::numeric_limits<double>::max();
    primal->setControlLowerBound(DISABLED_LOWER_BOUND);
    const double DISABLED_UPPER_BOUND = std::numeric_limits<double>::max();
    primal->setControlUpperBound(DISABLED_UPPER_BOUND);

    // Set objective function and equality constraint operators
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, m_ProblemType));
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality(new dotk::DOTk_MexEqualityConstraint(m_EqualityConstraint, m_ProblemType));

    // Set trust region algorithm data manager
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data(new dotk::DOTk_SteihaugTointDataMng(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], data);

    // Set numerically differentiated Hessian
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective, equality));
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    // Set trust region step manager
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step(new dotk::DOTk_KelleySachsStepMng(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    // Initialize trust region algorithm
    dotk::DOTk_SteihaugTointKelleySachs algorithm(data, step);
    this->setIxKelleySachsAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionKelleySachs::solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: control vectors
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);

    // Allocate DOTk data structures
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
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

    // Set trust region algorithm data manager
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, m_ProblemType));
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data(new dotk::DOTk_SteihaugTointDataMng(primal, objective));
    dotk::mex::buildGradient(input_[0], data);

    // Set numerically differentiated Hessian
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    // Set trust region step manager
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step(new dotk::DOTk_KelleySachsStepMng(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    // Initialize trust region algorithm
    dotk::DOTk_SteihaugTointKelleySachs algorithm(data, step);
    this->setIxKelleySachsAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionKelleySachs::solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    // Set objective function and equality constraint operators
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, m_ProblemType));
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality(new dotk::DOTk_MexEqualityConstraint(m_EqualityConstraint, m_ProblemType));

    // Set trust region algorithm data manager
    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data(new dotk::DOTk_SteihaugTointDataMng(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], data);

    // Set numerically differentiated Hessian
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective, equality));
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);

    // Set trust region step manager
    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step(new dotk::DOTk_KelleySachsStepMng(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    // Initialize trust region algorithm
    dotk::DOTk_SteihaugTointKelleySachs algorithm(data, step);
    this->setIxKelleySachsAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionKelleySachs::setIxKelleySachsAlgorithmParameters(dotk::DOTk_SteihaugTointKelleySachs & algorithm_)
{
    dotk::DOTk_MexSteihaugTointNewton::setAlgorithmParameters(algorithm_);

    size_t max_num_updates = this->getMaxNumUpdates();
    algorithm_.setMaxNumUpdates(max_num_updates);
    size_t max_num_solver_itr = this->getMaxNumSteihaugTointSolverItr();
    algorithm_.setMaxNumSolverItr(max_num_solver_itr);
    double actual_reduction_tolerance = dotk::DOTk_MexSteihaugTointNewton::getActualReductionTolerance();
    algorithm_.setActualReductionTolerance(actual_reduction_tolerance);

    algorithm_.printDiagnosticsEveryItrAndSolutionAtTheEnd();
}

}

