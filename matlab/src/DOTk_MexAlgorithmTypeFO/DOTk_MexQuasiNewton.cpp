/*
 * DOTk_MexQuasiNewton.cpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Dual.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_MexQuasiNewton.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexQuasiNewtonParser.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_LineSearchQuasiNewton.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeFO.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeFO.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

namespace dotk
{

DOTk_MexQuasiNewton::DOTk_MexQuasiNewton(const mxArray* options_[]) :
        dotk::DOTk_MexAlgorithmTypeFO(options_[0]),
        m_ObjectiveFunctionOperators(NULL),
        m_EqualityConstraintOperators(NULL)
{
    this->initialize(options_);
}

DOTk_MexQuasiNewton::~DOTk_MexQuasiNewton()
{
    this->clear();
}

void DOTk_MexQuasiNewton::solve(const mxArray* input_[], mxArray* output_[])
{
    DOTk_MexAlgorithmTypeFO::setNumControls(input_[0]);
    dotk::types::problem_t type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();

    switch (type)
    {
        case dotk::types::TYPE_ULP:
        {
            this->solveTypeLinearProgramming(input_, output_);
            break;
        }
        case dotk::types::TYPE_UNLP:
        {
            DOTk_MexAlgorithmTypeFO::setNumDuals(input_[0]);
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
            DOTk_MexAlgorithmTypeFO::setNumDuals(input_[0]);
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
            std::string msg(" DOTk/MEX ERROR: Invalid Problem Type for Quasi-Newton Method. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexQuasiNewton::solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();

    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    dotk::types::line_search_t step_type = this->getLineSearchMethod();
    step->build(primal, step_type);
    DOTk_MexAlgorithmTypeFO::setLineSearchMethodParameters(*step);

    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
        mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::mex::buildGradient(input_[0], primal, mng);

    this->optimize(step, mng, input_, output_);
}

void DOTk_MexQuasiNewton::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraintOperators);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    dotk::types::line_search_t step_type = this->getLineSearchMethod();
    step->build(primal, step_type);
    DOTk_MexAlgorithmTypeFO::setLineSearchMethodParameters(*step);

    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeUNP>
        mng(new dotk::DOTk_LineSearchMngTypeUNP(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], primal, mng);

    this->optimize(step, mng, input_, output_);
}

void DOTk_MexQuasiNewton::solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr< dotk::vector<double> > bounds = primal->control()->clone();
    dotk::mex::parseControlLowerBound(input_[0], *bounds);
    primal->setControlLowerBound(*bounds);

    bounds->fill(0.);
    dotk::mex::parseControlUpperBound(input_[0], *bounds);
    primal->setControlUpperBound(*bounds);

    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    dotk::types::line_search_t step_type = this->getLineSearchMethod();
    step->build(primal, step_type);
    DOTk_MexAlgorithmTypeFO::setLineSearchMethodParameters(*step);
    DOTk_MexAlgorithmTypeFO::setBoundConstraintMethod(input_[0], primal, step);

    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::mex::buildGradient(input_[0], primal, mng);

    this->optimize(step, mng, input_, output_);
}

void DOTk_MexQuasiNewton::solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraintOperators);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr< dotk::vector<double> > bounds = primal->control()->clone();
    dotk::mex::parseControlLowerBound(input_[0], *bounds);
    primal->setControlLowerBound(*bounds);

    bounds->fill(0.);
    dotk::mex::parseControlUpperBound(input_[0], *bounds);
    primal->setControlUpperBound(*bounds);

    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeUNP>
        mng(new dotk::DOTk_LineSearchMngTypeUNP(primal, objective, equality));

    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    dotk::types::line_search_t step_type = this->getLineSearchMethod();
    step->build(primal, step_type);
    DOTk_MexAlgorithmTypeFO::setLineSearchMethodParameters(*step);
    DOTk_MexAlgorithmTypeFO::setBoundConstraintMethod(input_[0], primal, step);

    dotk::mex::buildGradient(input_[0], primal, mng);

    this->optimize(step, mng, input_, output_);
}

void DOTk_MexQuasiNewton::setAlgorithmParameters(dotk::DOTk_LineSearchQuasiNewton & algorithm_)
{
    size_t max_num_itr = DOTk_MexAlgorithmTypeFO::getMaxNumAlgorithmItr();
    algorithm_.setMaxNumItr(max_num_itr);
    Real optimality_tolerance = DOTk_MexAlgorithmTypeFO::getOptimalityTolerance();
    algorithm_.setObjectiveFuncTol(optimality_tolerance);
    Real gradient_tolerance = DOTk_MexAlgorithmTypeFO::getGradientTolerance();
    algorithm_.setGradientTol(gradient_tolerance);
    Real trial_step_tolerance = DOTk_MexAlgorithmTypeFO::getTrialStepTolerance();
    algorithm_.setTrialStepTol(trial_step_tolerance);
}

void DOTk_MexQuasiNewton::clear()
{
    m_ObjectiveFunctionOperators.release();
    m_EqualityConstraintOperators.release();
}

void DOTk_MexQuasiNewton::initialize(const mxArray* options_[])
{
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunctionOperators);
}

void DOTk_MexQuasiNewton::optimize(const std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                                   const std::tr1::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_,
                                   const mxArray* input_[],
                                   mxArray* output_[])
{
    dotk::DOTk_LineSearchQuasiNewton algorithm(step_, mng_);

    dotk::mex::buildQuasiNewtonMethod(input_[0], algorithm);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();
    algorithm.getMin();

    DOTk_MexAlgorithmTypeFO::gatherOutputData(algorithm, *mng_, output_);
}

}
