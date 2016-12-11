/*
 * DOTk_MexGradientProjection.cpp
 *
 *  Created on: Oct 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexNonlinearCG.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexGradientProjection.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_GradientProjectionMethod.hpp"

namespace dotk
{

DOTk_MexGradientProjection::DOTk_MexGradientProjection(const mxArray* options_[]) :
        m_MaxNumIterations(5000),
        m_MaxNumLineSearchIterations(10),
        m_ObjectiveTolerance(1e-8),
        m_ProjectedGradientTolerance(1e-8),
        m_LineSearchContractionFactor(0.5),
        m_LineSearchStagnationTolerance(1e-8),
        m_ProblemType(dotk::types::problem_t::PROBLEM_TYPE_UNDEFINED),
        m_LineSearchMethod(dotk::types::LINE_SEARCH_DISABLED),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr)
{
    this->initialize(options_);
}

DOTk_MexGradientProjection::~DOTk_MexGradientProjection()
{
    this->clear();
}

void DOTk_MexGradientProjection::clear()
{
    m_ObjectiveFunction.release();
    m_EqualityConstraint.release();
}

void DOTk_MexGradientProjection::initialize(const mxArray* options_[])
{
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunction);

    dotk::mex::parseProblemType(options_[0], m_ProblemType);
    dotk::mex::parseLineSearchMethod(options_[0], m_LineSearchMethod);
    dotk::mex::parseMaxNumAlgorithmItr(options_[0], m_MaxNumIterations);
    dotk::mex::parseObjectiveTolerance(options_[0], m_ObjectiveTolerance);
    dotk::mex::parseGradientTolerance(options_[0], m_ProjectedGradientTolerance);

    dotk::mex::parseMaxNumLineSearchItr(options_[0], m_MaxNumLineSearchIterations);
    dotk::mex::parseLineSearchContractionFactor(options_[0], m_LineSearchContractionFactor);
    dotk::mex::parseLineSearchStagnationTolerance(options_[0], m_LineSearchStagnationTolerance);
}

void DOTk_MexGradientProjection::solve(const mxArray* input_[], mxArray* output_[])
{
    switch(m_ProblemType)
    {
        case dotk::types::TYPE_ULP:
        {
            this->solveLinearProgrammingProblem(input_, output_);
            break;
        }
        case dotk::types::TYPE_UNLP:
        {
            this->solveNonlinearProgrammingProblem(input_, output_);
            break;
        }
        case dotk::types::TYPE_CLP:
        case dotk::types::TYPE_CNLP:
        case dotk::types::TYPE_LP_BOUND:
        case dotk::types::TYPE_NLP_BOUND:
        case dotk::types::TYPE_ELP:
        case dotk::types::TYPE_ENLP:
        case dotk::types::TYPE_ELP_BOUND:
        case dotk::types::TYPE_ENLP_BOUND:
        case dotk::types::TYPE_ILP:
        case dotk::types::PROBLEM_TYPE_UNDEFINED:
        default:
        {
            std::string msg("\n\n DOTk/MEX ERROR: Invalid Problem Type for Gradient Projection Method. See Users' Manual. \n\n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexGradientProjection::solveLinearProgrammingProblem(const mxArray* input_[], mxArray* output_[])
{
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr< dotk::Vector<double> > vector = primal->control()->clone();
    dotk::mex::parseControlLowerBound(input_[0], *vector);
    primal->setControlLowerBound(*vector);
    vector->fill(0.);
    dotk::mex::parseControlUpperBound(input_[0], *vector);
    primal->setControlUpperBound(*vector);

    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->build(primal, m_LineSearchMethod);
    step->setMaxNumIterations(m_MaxNumLineSearchIterations);
    step->setContractionFactor(m_LineSearchContractionFactor);
    step->setStagnationTolerance(m_LineSearchStagnationTolerance);

    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunction.get(), m_ProblemType));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
        mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::mex::buildGradient(input_[0], primal, mng);

    dotk::GradientProjectionMethod algorithm(primal, step, mng);
    algorithm.setMaxNumIterations(m_MaxNumIterations);
    algorithm.setObjectiveTolerance(m_ObjectiveTolerance);
    algorithm.setProjectedGradientTolerance(m_ProjectedGradientTolerance);

    algorithm.printDiagnostics();
    algorithm.getMin();

    this->outputData(algorithm, *mng, output_);
}

void DOTk_MexGradientProjection::solveNonlinearProgrammingProblem(const mxArray* input_[], mxArray* output_[])
{
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr< dotk::Vector<double> > vector = primal->control()->clone();
    dotk::mex::parseControlLowerBound(input_[0], *vector);
    primal->setControlLowerBound(*vector);
    vector->fill(0.);
    dotk::mex::parseControlUpperBound(input_[0], *vector);
    primal->setControlUpperBound(*vector);

    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->build(primal, m_LineSearchMethod);
    step->setMaxNumIterations(m_MaxNumLineSearchIterations);
    step->setContractionFactor(m_LineSearchContractionFactor);
    step->setStagnationTolerance(m_LineSearchStagnationTolerance);

    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunction.get(), m_ProblemType));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraint);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraint.get(), m_ProblemType));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeUNP>
        mng(new dotk::DOTk_LineSearchMngTypeUNP(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], primal, mng);

    dotk::GradientProjectionMethod algorithm(primal, step, mng);
    algorithm.setMaxNumIterations(m_MaxNumIterations);
    algorithm.setObjectiveTolerance(m_ObjectiveTolerance);
    algorithm.setProjectedGradientTolerance(m_ProjectedGradientTolerance);

    algorithm.printDiagnostics();
    algorithm.getMin();

    this->outputData(algorithm, *mng, output_);
}

void DOTk_MexGradientProjection::outputData(const dotk::GradientProjectionMethod & algorithm_,
                                            const dotk::DOTk_LineSearchAlgorithmsDataMng & mng_,
                                            mxArray* output_[])
{
    // Create memory allocation for output struct
    const char *field_names[7] =
        { "Iterations", "ObjectiveFunctionValue", "Control", "Gradient", "NormGradient", "ProjectedGradient", "NormProjectedGradient" };
    output_[0] = mxCreateStructMatrix(1, 1, 7, field_names);

    dotk::DOTk_MexArrayPtr iteration_count(mxCreateNumericMatrix_730(1, 1, mxINDEX_CLASS, mxREAL));
    static_cast<size_t*>(mxGetData(iteration_count.get()))[0] = algorithm_.getIterationCount();
    mxSetField(output_[0], 0, "Iterations", iteration_count.get());

    dotk::DOTk_MexArrayPtr objective_function_value(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(objective_function_value.get())[0] = mng_.getNewObjectiveFunctionValue();
    mxSetField(output_[0], 0, "ObjectiveFunctionValue", objective_function_value.get());

    size_t num_controls = mng_.getNewPrimal()->size();
    dotk::DOTk_MexArrayPtr primal(mxCreateDoubleMatrix(num_controls, 1, mxREAL));
    mng_.getNewPrimal()->gather(mxGetPr(primal.get()));
    mxSetField(output_[0], 0, "Control", primal.get());

    dotk::DOTk_MexArrayPtr gradient(mxCreateDoubleMatrix(num_controls, 1, mxREAL));
    mng_.getNewGradient()->gather(mxGetPr(gradient.get()));
    mxSetField(output_[0], 0, "Gradient", gradient.get());

    dotk::DOTk_MexArrayPtr norm_gradient(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_gradient.get())[0] = mng_.getNewGradient()->norm();
    mxSetField(output_[0], 0, "NormGradient", norm_gradient.get());

    dotk::DOTk_MexArrayPtr projected_gradient(mxCreateDoubleMatrix(num_controls, 1, mxREAL));
    mng_.getTrialStep()->gather(mxGetPr(projected_gradient.get()));
    mxSetField(output_[0], 0, "ProjectedGradient", projected_gradient.get());

    dotk::DOTk_MexArrayPtr norm_projected_gradient(mxCreateDoubleMatrix(1, 1, mxREAL));
    mxGetPr(norm_projected_gradient.get())[0] = mng_.getTrialStep()->norm();
    mxSetField(output_[0], 0, "NormProjectedGradient", norm_projected_gradient.get());

    iteration_count.release();
    objective_function_value.release();
    primal.release();
    gradient.release();
    norm_gradient.release();
    projected_gradient.release();
    norm_projected_gradient.release();
}

}
