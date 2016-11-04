/*
 * DOTk_MexNewtonTypeTR.cpp
 *
 *  Created on: Apr 24, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexNewtonTypeTR.hpp"
#include "DOTk_MexHessianFactory.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_TrustRegionMngTypeULP.hpp"
#include "DOTk_TrustRegionMngTypeUNP.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_TrustRegionInexactNewton.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"
#include "DOTk_TrustRegionAlgorithmsDataMng.hpp"

namespace dotk
{

DOTk_MexNewtonTypeTR::DOTk_MexNewtonTypeTR(const mxArray* options_[]) :
        dotk::DOTk_MexAlgorithmTypeNewton(options_[0]),
        m_MaxNumTrustRegionSubProblemItr(50),
        m_MaxTrustRegionRadius(1e4),
        m_MinTrustRegionRadius(1e-6),
        m_InitialTrustRegionRadius(1e3),
        m_TrustRegionExpansionFactor(2),
        m_TrustRegionContractionFactor(0.5),
        m_MinActualOverPredictedReductionRatio(0.25)
{
    this->initialize(options_);
}

DOTk_MexNewtonTypeTR::~DOTk_MexNewtonTypeTR()
{
    this->clear();
}

size_t DOTk_MexNewtonTypeTR::getMaxNumTrustRegionSubProblemItr() const
{
    return (m_MaxNumTrustRegionSubProblemItr);
}

double DOTk_MexNewtonTypeTR::getMaxTrustRegionRadius() const
{
    return (m_MaxTrustRegionRadius);
}

double DOTk_MexNewtonTypeTR::getMinTrustRegionRadius() const
{
    return (m_MinTrustRegionRadius);
}

double DOTk_MexNewtonTypeTR::getInitialTrustRegionRadius() const
{
    return (m_InitialTrustRegionRadius);
}

double DOTk_MexNewtonTypeTR::getTrustRegionExpansionFactor() const
{
    return (m_TrustRegionExpansionFactor);
}

double DOTk_MexNewtonTypeTR::getTrustRegionContractionFactor() const
{
    return (m_TrustRegionContractionFactor);
}

double DOTk_MexNewtonTypeTR::getMinActualOverPredictedReductionRatio() const
{
    return (m_MinActualOverPredictedReductionRatio);
}

void DOTk_MexNewtonTypeTR::solve(const mxArray* input_[], mxArray* output_[])
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
            std::string msg(" DOTk/MEX ERROR: Invalid Problem Type for Trust Region Based Newton Method. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexNewtonTypeTR::solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeNewton::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeULP>
        mng(new dotk::DOTk_TrustRegionMngTypeULP(primal, objective));
    dotk::mex::buildGradient(input_[0], primal, mng);
    dotk::mex::buildTrustRegionMethod(input_[0], mng);
    this->setTrustRegionMethodParameters(mng);
    this->optimize(mng, input_, output_);
}

void DOTk_MexNewtonTypeTR::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeNewton::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraintOperators);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeUNP>
        mng(new dotk::DOTk_TrustRegionMngTypeUNP(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], primal, mng);
    dotk::mex::buildTrustRegionMethod(input_[0], mng);
    this->setTrustRegionMethodParameters(mng);
    this->optimize(mng, input_, output_);
}

void DOTk_MexNewtonTypeTR::setAlgorithmParameters(dotk::DOTk_TrustRegionInexactNewton & algorithm_)
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

void DOTk_MexNewtonTypeTR::setTrustRegionMethodParameters
(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_)
{
    double value = this->getMaxTrustRegionRadius();
    mng_->setMaxTrustRegionRadius(value);
    value = this->getMinTrustRegionRadius();
    mng_->setMinTrustRegionRadius(value);
    value = this->getInitialTrustRegionRadius();
    mng_->setTrustRegionRadius(value);
    value = this->getTrustRegionExpansionFactor();
    mng_->setTrustRegionExpansionParameter(value);
    value = this->getTrustRegionContractionFactor();
    mng_->setTrustRegionContractionParameter(value);
    size_t itr = this->getMaxNumTrustRegionSubProblemItr();
    mng_->setMaxTrustRegionSubProblemIterations(itr);
    value = this->getMinActualOverPredictedReductionRatio();
    mng_->setMinActualOverPredictedReductionAllowed(value);
}

void DOTk_MexNewtonTypeTR::clear()
{
    m_ObjectiveFunctionOperators.release();
    m_EqualityConstraintOperators.release();
}

void DOTk_MexNewtonTypeTR::initialize(const mxArray* options_[])
{
    dotk::mex::parseMaxTrustRegionRadius(options_[0], m_MaxTrustRegionRadius);
    dotk::mex::parseMinTrustRegionRadius(options_[0], m_MinTrustRegionRadius);
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunctionOperators);
    dotk::mex::parseInitialTrustRegionRadius(options_[0], m_InitialTrustRegionRadius);
    dotk::mex::parseTrustRegionExpansionFactor(options_[0], m_TrustRegionExpansionFactor);
    dotk::mex::parseTrustRegionContractionFactor(options_[0], m_TrustRegionContractionFactor);
    dotk::mex::parseMaxNumTrustRegionSubProblemItr(options_[0], m_MaxNumTrustRegionSubProblemItr);
    dotk::mex::parseMinActualOverPredictedReductionRatio(options_[0], m_MinActualOverPredictedReductionRatio);
}

void DOTk_MexNewtonTypeTR::optimize(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_,
                                    const mxArray* input_[],
                                    mxArray* output_[])
{
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildControlContainer(input_[0], *primal);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::mex::buildHessian(input_[0], primal, hessian);

    dotk::DOTk_TrustRegionInexactNewton algorithm(hessian, mng_);
    dotk::mex::buildKrylovSolver(input_[0], primal, algorithm);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    algorithm.getMin();

    DOTk_MexAlgorithmTypeNewton::gatherOutputData(algorithm, *mng_, output_);
}

}
