/*
 * DOTk_MexOptimalityCriteria.cpp
 *
 *  Created on: Jun 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <sstream>
#include <fstream>
#include <tr1/memory>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_OptimalityCriteria.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexOptimalityCriteria.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexInequalityConstraint.cpp"
#include "DOTk_MexInequalityConstraint.hpp"
#include "DOTk_MexOptimalityCriteriaParser.hpp"

namespace dotk
{

DOTk_MexOptimalityCriteria::DOTk_MexOptimalityCriteria(const mxArray* options_[]) :
        m_MaxNumAlgorithmItr(0),
        m_MoveLimit(0),
        m_DualLowerBound(0),
        m_DualUpperBound(0),
        m_DampingParameter(0),
        m_GradientTolerance(0),
        m_BisectionTolerance(0),
        m_OptimalityTolerance(0),
        m_FeasibilityTolerance(0),
        m_ControlStagnationTolerance(0),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED),
        m_ObjectiveFunctionOperators(NULL),
        m_EqualityConstraintOperators(NULL),
        m_InequalityConstraintOperators(NULL)
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
    return (m_OptimalityTolerance);
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
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildPrimalContainer(input_[0], *primal);

    std::tr1::shared_ptr< dotk::Vector<double> > vector = primal->control()->clone();
    dotk::mex::parseControlLowerBound(input_[0], *vector);
    primal->setControlLowerBound(*vector);

    vector->fill(0.);
    dotk::mex::parseControlUpperBound(input_[0], *vector);
    primal->setControlUpperBound(*vector);

    dotk::types::problem_t type = this->getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint<double> >
        inequality(new dotk::DOTk_MexInequalityConstraint<double>(m_InequalityConstraintOperators.get(), type));

    dotk::DOTk_OptimalityCriteria algorithm(primal, objective, equality, inequality);
    this->setAlgorithmParameters(input_[0], algorithm);
    algorithm.enableDiagnostics();
    algorithm.getMin();

    this->printOutputFile(algorithm);
    this->gatherOutputData(algorithm, *vector, output_);
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

void DOTk_MexOptimalityCriteria::gatherOutputData(dotk::DOTk_OptimalityCriteria & algorithm_,
                                                  dotk::Vector<double> & solution_,
                                                  mxArray* output_[])
{
    // Create memory allocation for output struct
    const char *field_names[7] =
        { "Iterations", "ObjectiveFunctionValue", "Control", "Gradient", "NormGradient", "Dual",
                "InequalityConstraintResidual" };
    output_[0] = mxCreateStructMatrix(1, 1, 7, field_names);

    dotk::DOTk_MexArrayPtr iterations(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(iterations.get()))[0] = algorithm_.getNumItrDone();
    mxSetField(output_[0], 0, "Iterations", iterations.get());

    dotk::DOTk_MexArrayPtr optimality(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(optimality.get())[0] = algorithm_.getOptimalObjectiveFunctionValue();
    mxSetField(output_[0], 0, "ObjectiveFunctionValue", optimality.get());

    size_t num_controls = solution_.size();
    dotk::DOTk_MexArrayPtr solution(mxCreateDoubleMatrix(num_controls, 1, mxREAL));
    algorithm_.gatherSolution(solution_);
    solution_.gather(mxGetPr(solution.get()));
    mxSetField(output_[0], 0, "Control", solution.get());

    dotk::DOTk_MexArrayPtr gradient(mxCreateDoubleMatrix(num_controls, 1, mxREAL));
    algorithm_.gatherGradient(solution_);
    solution_.gather(mxGetPr(gradient.get()));
    mxSetField(output_[0], 0, "Gradient", gradient.get());

    dotk::DOTk_MexArrayPtr norm_gradient(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_gradient.get())[0] = solution_.norm();
    mxSetField(output_[0], 0, "NormGradient", norm_gradient.get());

    dotk::DOTk_MexArrayPtr inequality_constraint_dual(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(inequality_constraint_dual.get())[0] = algorithm_.getInequalityDual();
    mxSetField(output_[0], 0, "Dual", inequality_constraint_dual.get());

    dotk::DOTk_MexArrayPtr inequality_constraint_residual(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(inequality_constraint_residual.get())[0] = algorithm_.getInequalityConstraintResidual();
    mxSetField(output_[0], 0, "InequalityConstraintResidual", inequality_constraint_residual.get());

    iterations.release();
    optimality.release();
    solution.release();
    gradient.release();
    norm_gradient.release();
    inequality_constraint_dual.release();
    inequality_constraint_residual.release();
}

void DOTk_MexOptimalityCriteria::clear()
{
    m_ObjectiveFunctionOperators.release();
    m_EqualityConstraintOperators.release();
    m_InequalityConstraintOperators.release();
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
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunctionOperators);
    dotk::mex::parseEqualityConstraint(options_[1], m_EqualityConstraintOperators);
    dotk::mex::parseInequalityConstraint(options_[1], m_InequalityConstraintOperators);

    dotk::mex::parseProblemType(options_[0], m_ProblemType);
    dotk::mex::parseGradientTolerance(options_[0], m_GradientTolerance);
    dotk::mex::parseMaxNumAlgorithmItr(options_[0], m_MaxNumAlgorithmItr);
    dotk::mex::parseOptimalityTolerance(options_[0], m_OptimalityTolerance);
    dotk::mex::parseFeasibilityTolerance(options_[0], m_FeasibilityTolerance);
    dotk::mex::parseControlStagnationTolerance(options_[0], m_ControlStagnationTolerance);

    dotk::mex::parseOptCriteriaMoveLimit(options_[0], m_MoveLimit);
    dotk::mex::parseOptCriteriaDualLowerBound(options_[0], m_DualLowerBound);
    dotk::mex::parseOptCriteriaDualUpperBound(options_[0], m_DualUpperBound);
    dotk::mex::parseOptCriteriaDampingParameter(options_[0], m_DampingParameter);
    dotk::mex::parseOptCriteriaBisectionTolerance(options_[0], m_BisectionTolerance);
}

}
