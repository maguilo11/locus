/*
 * DOTk_MexTrustRegionLinMore.cpp
 *
 *  Created on: Apr 17, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_MexHessianFactory.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_SteihaugTointStepMng.hpp"
#include "DOTk_SteihaugTointDataMng.hpp"
#include "DOTk_SteihaugTointLinMore.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexKrylovSolverParser.hpp"
#include "DOTk_MexTrustRegionLinMore.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_SteihaugTointProjGradStep.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

namespace dotk
{

DOTk_MexTrustRegionLinMore::DOTk_MexTrustRegionLinMore(const mxArray* options_[]) :
        dotk::DOTk_MexSteihaugTointNewton(options_),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED),
        m_MaxNumSteihaugTointSolverItr(200),
        m_SolverRelativeTolerance(0.1),
        m_SolverRelativeToleranceExponential(0.5)
{
    this->initializeTrustRegionLinMore(options_);
}

DOTk_MexTrustRegionLinMore::~DOTk_MexTrustRegionLinMore()
{
    this->clear();
}

void DOTk_MexTrustRegionLinMore::clear()
{
    m_ObjectiveFunctionOperators.release();
    m_EqualityConstraintOperators.release();
}

void DOTk_MexTrustRegionLinMore::initializeTrustRegionLinMore(const mxArray* options_[])
{
    dotk::mex::parseProblemType(options_[0], m_ProblemType);
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunctionOperators);
    dotk::mex::parseMaxNumKrylovSolverItr(options_[0], m_MaxNumSteihaugTointSolverItr);
    dotk::mex::parseKrylovSolverRelativeTolerance(options_[0], m_SolverRelativeTolerance);
    dotk::mex::parseRelativeToleranceExponential(options_[0], m_SolverRelativeToleranceExponential);
}

size_t DOTk_MexTrustRegionLinMore::getMaxNumSteihaugTointSolverItr() const
{
    return (m_MaxNumSteihaugTointSolverItr);
}

double DOTk_MexTrustRegionLinMore::getSolverRelativeTolerance() const
{
    return (m_SolverRelativeTolerance);
}

double DOTk_MexTrustRegionLinMore::getSolverRelativeToleranceExponential() const
{
    return (m_SolverRelativeToleranceExponential);
}

void DOTk_MexTrustRegionLinMore::solve(const mxArray* input_[], mxArray* output_[])
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
            std::string msg(" DOTk/MEX ERROR: Invalid Problem Type for Lin-More Trust Region Newton Algorithm. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexTrustRegionLinMore::solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), m_ProblemType));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data(new dotk::DOTk_SteihaugTointDataMng(primal, objective));
    dotk::mex::buildGradient(input_[0], primal, data);

    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::mex::buildHessian(input_[0], primal, hessian);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng>
        step(new dotk::DOTk_SteihaugTointStepMng(primal, hessian));
   dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    dotk::DOTk_SteihaugTointLinMore algorithm(data, step);
    this->setLinMoreAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexTrustRegionLinMore::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), m_ProblemType));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraintOperators);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), m_ProblemType));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointDataMng>
        data(new dotk::DOTk_SteihaugTointDataMng(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], primal, data);

    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::mex::buildHessian(input_[0], primal, hessian);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointStepMng>
        step(new dotk::DOTk_SteihaugTointStepMng(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    dotk::DOTk_SteihaugTointLinMore algorithm(data, step);
    this->setLinMoreAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexTrustRegionLinMore::solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::mex::buildHessian(input_[0], primal, hessian);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step(new dotk::DOTk_SteihaugTointProjGradStep(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    dotk::DOTk_SteihaugTointLinMore algorithm(data, step);
    this->setLinMoreAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexTrustRegionLinMore::solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::mex::buildHessian(input_[0], primal, hessian);

    std::tr1::shared_ptr<dotk::DOTk_SteihaugTointProjGradStep>
        step(new dotk::DOTk_SteihaugTointProjGradStep(primal, hessian));
    dotk::DOTk_MexSteihaugTointNewton::setTrustRegionStepParameters(step);

    dotk::DOTk_SteihaugTointLinMore algorithm(data, step);
    this->setLinMoreAlgorithmParameters(algorithm);

    algorithm.getMin();

    dotk::DOTk_MexSteihaugTointNewton::gatherOutputData(algorithm, *data, *step, output_);
}

void DOTk_MexTrustRegionLinMore::setLinMoreAlgorithmParameters(dotk::DOTk_SteihaugTointLinMore & algorithm_)
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
