/*
 * DOTk_MexIxTrustRegionKelleySachs.cpp
 *
 *  Created on: Apr 17, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_KelleySachsStepMng.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_SteihaugTointDataMng.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexKrylovSolverParser.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_SteihaugTointKelleySachs.hpp"
#include "DOTk_MexNumDiffHessianFactory.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"
#include "DOTk_MexIxTrustRegionKelleySachs.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace dotk
{

DOTk_MexIxTrustRegionKelleySachs::DOTk_MexIxTrustRegionKelleySachs(const mxArray* options_[]) :
        dotk::DOTk_MexSteihaugTointNewton(options_),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED),
        m_MaxNumUpdates(10),
        m_MaxNumSteihaugTointSolverItr(200)
{
    this->initializeIxKelleySachsTrustRegion(options_);
}

DOTk_MexIxTrustRegionKelleySachs::~DOTk_MexIxTrustRegionKelleySachs()
{
    this->clear();
}

void DOTk_MexIxTrustRegionKelleySachs::clear()
{
    m_ObjectiveFunctionOperators.release();
    m_EqualityConstraintOperators.release();
}

void DOTk_MexIxTrustRegionKelleySachs::initializeIxKelleySachsTrustRegion(const mxArray* options_[])
{
    dotk::mex::parseProblemType(options_[0], m_ProblemType);
    dotk::mex::parseMaxNumUpdates(options_[0], m_MaxNumUpdates);
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunctionOperators);
    dotk::mex::parseMaxNumKrylovSolverItr(options_[0], m_MaxNumSteihaugTointSolverItr);
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
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), m_ProblemType));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildControlContainer(input_[0], *primal);
    const double DISABLED_LOWER_BOUND = -std::numeric_limits<Real>::max();
    primal->setControlLowerBound(DISABLED_LOWER_BOUND);
    const double DISABLED_UPPER_BOUND = std::numeric_limits<Real>::max();
    primal->setControlUpperBound(DISABLED_UPPER_BOUND);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
    data(new dotk::DOTk_SteihaugTointDataMng(primal, objective));
    dotk::mex::buildGradient(input_[0], primal, data);

    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], primal, hessian);

    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step(new dotk::DOTk_KelleySachsStepMng(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    dotk::DOTk_SteihaugTointKelleySachs algorithm(data, step);
    this->setIxKelleySachsAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionKelleySachs::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), m_ProblemType));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraintOperators);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), m_ProblemType));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);
    const double DISABLED_LOWER_BOUND = -std::numeric_limits<Real>::max();
    primal->setControlLowerBound(DISABLED_LOWER_BOUND);
    const double DISABLED_UPPER_BOUND = std::numeric_limits<Real>::max();
    primal->setControlUpperBound(DISABLED_UPPER_BOUND);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data(new dotk::DOTk_SteihaugTointDataMng(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], primal, data);

    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective, equality));
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], primal, hessian);

    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step(new dotk::DOTk_KelleySachsStepMng(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    dotk::DOTk_SteihaugTointKelleySachs algorithm(data, step);
    this->setIxKelleySachsAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionKelleySachs::solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), m_ProblemType));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::vector<double> > bounds = primal->control()->clone();
    dotk::mex::parseControlLowerBound(input_[0], *bounds);
    primal->setControlLowerBound(*bounds);
    bounds->fill(0.);
    dotk::mex::parseControlUpperBound(input_[0], *bounds);
    primal->setControlUpperBound(*bounds);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data(new dotk::DOTk_SteihaugTointDataMng(primal, objective));
    dotk::mex::buildGradient(input_[0], primal, data);

    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], primal, hessian);

    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step(new dotk::DOTk_KelleySachsStepMng(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    dotk::DOTk_SteihaugTointKelleySachs algorithm(data, step);
    this->setIxKelleySachsAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexIxTrustRegionKelleySachs::solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), m_ProblemType));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraintOperators);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), m_ProblemType));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::vector<double> > bounds = primal->control()->clone();
    dotk::mex::parseControlLowerBound(input_[0], *bounds);
    primal->setControlLowerBound(*bounds);
    bounds->fill(0.);
    dotk::mex::parseControlUpperBound(input_[0], *bounds);
    primal->setControlUpperBound(*bounds);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data(new dotk::DOTk_SteihaugTointDataMng(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], primal, data);

    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
        hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective, equality));
    dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], primal, hessian);

    std::tr1::shared_ptr<dotk::DOTk_KelleySachsStepMng> step(new dotk::DOTk_KelleySachsStepMng(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

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

