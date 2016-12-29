/*
 * DOTk_MexOptimalityCriteria.cpp
 *
 *  Created on: Jun 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>
#include <fstream>
#include <tr1/memory>

#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexOptimalityCriteria.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexInequalityConstraint.hpp"
#include "DOTk_MexOptimalityCriteriaParser.hpp"

#include "DOTk_Primal.hpp"
#include "DOTk_OptimalityCriteria.hpp"

namespace dotk
{

DOTk_MexOptimalityCriteria::DOTk_MexOptimalityCriteria(const mxArray* options_[]) :
        m_NumberControls(0),
        m_MaxNumAlgorithmItr(0),
        m_MoveLimit(0),
        m_DualLowerBound(0),
        m_DualUpperBound(0),
        m_DampingParameter(0),
        m_GradientTolerance(0),
        m_BisectionTolerance(0),
        m_ObjectiveTolerance(0),
        m_FeasibilityTolerance(0),
        m_ControlStagnationTolerance(0),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr),
        m_InequalityConstraint(nullptr)
{
    this->initialize(options_);
}

DOTk_MexOptimalityCriteria::~DOTk_MexOptimalityCriteria()
{
    this->clear();
}

size_t DOTk_MexOptimalityCriteria::getMaxNumAlgorithmItr() const
{
    return (m_MaxNumAlgorithmItr);
}

double DOTk_MexOptimalityCriteria::getMoveLimit() const
{
    return (m_MoveLimit);
}

double DOTk_MexOptimalityCriteria::getDualLowerBound() const
{
    return (m_DualLowerBound);
}

double DOTk_MexOptimalityCriteria::getDualUpperBound() const
{
    return (m_DualUpperBound);
}

double DOTk_MexOptimalityCriteria::getDampingParameter() const
{
    return (m_DampingParameter);
}

double DOTk_MexOptimalityCriteria::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

double DOTk_MexOptimalityCriteria::getBisectionTolerance() const
{
    return (m_BisectionTolerance);
}

double DOTk_MexOptimalityCriteria::getFeasibilityTolerance() const
{
    return (m_FeasibilityTolerance);
}

double DOTk_MexOptimalityCriteria::getObjectiveFunctionTolerance() const
{
    return (m_ObjectiveTolerance);
}

double DOTk_MexOptimalityCriteria::getControlStagnationTolerance() const
{
    return (m_ControlStagnationTolerance);
}

dotk::types::problem_t DOTk_MexOptimalityCriteria::getProblemType() const
{
    return (m_ProblemType);
}

void DOTk_MexOptimalityCriteria::solve(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: state, control, and dual vectors
    size_t num_duals = 1;
    dotk::MexVector duals(num_duals, 0.);
    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector states(num_states, 0.);
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    m_NumberControls = controls.size();
    mxDestroyArray(mx_initial_control);

    // Allocate DOTk data structures
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateUserDefinedDual(duals);
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

    // Set objective, equality, and inequality operators
    dotk::types::problem_t type = this->getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, type));
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality(new dotk::DOTk_MexEqualityConstraint(m_EqualityConstraint, type));
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint>
        inequality(new dotk::DOTk_MexInequalityConstraint(m_InequalityConstraint, type));

    // Set optimization algorithm
    dotk::DOTk_OptimalityCriteria algorithm(primal, objective, equality, inequality);
    this->setAlgorithmParameters(input_[0], algorithm);
    algorithm.enableDiagnostics();

    algorithm.getMin(); // solve optimization problem

    this->printOutputFile(algorithm);
    this->gatherOutputData(algorithm, output_);
}

void DOTk_MexOptimalityCriteria::setAlgorithmParameters(const mxArray* options_,
                                                        dotk::DOTk_OptimalityCriteria & algorithm_)
{
    double value = this->getMoveLimit();
    algorithm_.setMoveLimit(value);
    value = this->getDampingParameter();
    algorithm_.setDampingParameter(value);
    value = this->getGradientTolerance();
    algorithm_.setGradientTolerance(value);
    value = this->getBisectionTolerance();
    algorithm_.setBisectionTolerance(value);
    value = this->getFeasibilityTolerance();
    algorithm_.setFeasibilityTolerance(value);
    value = this->getDualLowerBound();
    algorithm_.setInequalityConstraintDualLowerBound(value);
    value = this->getDualUpperBound();
    algorithm_.setInequalityConstraintDualUpperBound(value);
    value = this->getControlStagnationTolerance();
    algorithm_.setControlStagnationTolerance(value);

    size_t max_num_itr  = this->getMaxNumAlgorithmItr();
    algorithm_.setMaxNumOptimizationItr(max_num_itr);
}

void DOTk_MexOptimalityCriteria::gatherOutputData(dotk::DOTk_OptimalityCriteria & algorithm_, mxArray* outputs_[])
{
    // Create memory allocation for output struct
    const char *field_names[6] =
        { "Iterations", "ObjectiveFunctionValue", "Control", "Gradient", "NormGradient", "InequalityResidual" };
    outputs_[0] = mxCreateStructMatrix(1, 1, 6, field_names);

    /* NOTE: mxSetField does not create a copy of the data allocated. Thus,
     * mxDestroyArray cannot be called. Furthermore, MEX array data (e.g.
     * control, gradient, etc.) should be duplicated since the data in the
     * manager will be deallocated at the end. */
    mxArray* mx_number_iterations = mxCreateNumericMatrix(1, 1, mxINDEX_CLASS, mxREAL);
    static_cast<size_t*>(mxGetData(mx_number_iterations))[0] = algorithm_.getNumItrDone();
    mxSetField(outputs_[0], 0, "Iterations", mx_number_iterations);

    double value = algorithm_.getOptimalObjectiveFunctionValue();
    mxArray* mx_objective_function_value = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "ObjectiveFunctionValue", mx_objective_function_value);

    dotk::MexVector work(m_NumberControls, 0.);
    algorithm_.gatherSolution(work);
    mxArray* mx_control = mxDuplicateArray(work.array());
    mxSetField(outputs_[0], 0, "Control", mx_control);

    algorithm_.gatherGradient(work);
    mxArray* mx_gradient = mxDuplicateArray(work.array());
    mxSetField(outputs_[0], 0, "Gradient", mx_gradient);

    value = work.norm();
    mxArray* mx_norm_gradient = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "NormGradient", mx_norm_gradient);

    value = algorithm_.getInequalityConstraintResidual();
    mxArray* mx_inequality_residual = mxCreateDoubleScalar(value);
    mxSetField(outputs_[0], 0, "InequalityResidual", mx_inequality_residual);
}

void DOTk_MexOptimalityCriteria::clear()
{
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
    dotk::mex::destroy(m_InequalityConstraint);
}

void DOTk_MexOptimalityCriteria::printOutputFile(dotk::DOTk_OptimalityCriteria & algorithm_)
{
    std::ofstream output_file;
    std::ostringstream output_stream;
    algorithm_.gatherOuputStream(output_stream);
    output_file.open("DOTk_OptimalityCriteriaDiagnostics.out", std::ios::out | std::ios::trunc);
    output_file << output_stream.str().c_str();
    output_file.close();
}

void DOTk_MexOptimalityCriteria::initialize(const mxArray* options_[])
{
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(options_[1]);
    m_InequalityConstraint = dotk::mex::parseInequalityConstraint(options_[1]);

    m_ProblemType = dotk::mex::parseProblemType(options_[0]);
    m_GradientTolerance = dotk::mex::parseGradientTolerance(options_[0]);
    m_ObjectiveTolerance = dotk::mex::parseObjectiveTolerance(options_[0]);
    m_MaxNumAlgorithmItr = dotk::mex::parseMaxNumOuterIterations(options_[0]);
    m_FeasibilityTolerance = dotk::mex::parseFeasibilityTolerance(options_[0]);
    m_ControlStagnationTolerance = dotk::mex::parseControlStagnationTolerance(options_[0]);

    m_MoveLimit = dotk::mex::parseOptCriteriaMoveLimit(options_[0]);
    m_DualLowerBound = dotk::mex::parseOptCriteriaDualLowerBound(options_[0]);
    m_DualUpperBound = dotk::mex::parseOptCriteriaDualUpperBound(options_[0]);
    m_DampingParameter = dotk::mex::parseOptCriteriaDampingParameter(options_[0]);
    m_BisectionTolerance = dotk::mex::parseOptCriteriaBisectionTolerance(options_[0]);
}

}
