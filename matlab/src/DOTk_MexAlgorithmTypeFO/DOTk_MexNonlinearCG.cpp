/*
 * DOTk_MexNonlinearCG.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */


#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_NonlinearCG.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_MexNonlinearCG.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"
#include "DOTk_MexNumDiffHessianFactory.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace dotk
{

DOTk_MexNonlinearCG::DOTk_MexNonlinearCG(const mxArray* options_[]) :
        dotk::DOTk_MexAlgorithmTypeFO(options_[0]),
        m_NonlinearCgType(dotk::types::UNDEFINED_NLCG),
        m_ObjectiveFunctionOperators(),
        m_EqualityConstraintOperators(NULL)
{
    this->initialize(options_);
}

DOTk_MexNonlinearCG::~DOTk_MexNonlinearCG()
{
    this->clear();
}

void DOTk_MexNonlinearCG::solve(const mxArray* input_[], mxArray* output_[])
{
    DOTk_MexAlgorithmTypeFO::setNumControls(input_[0]);
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeFO::getProblemType();

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
            std::string msg(" DOTk/MEX ERROR: Invalid Problem Type for Nonlinear Conjugate Gradient Method. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexNonlinearCG::clear()
{
    m_ObjectiveFunctionOperators.release();
    m_EqualityConstraintOperators.release();
}

void DOTk_MexNonlinearCG::initialize(const mxArray* options_[])
{
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunctionOperators);
}

void DOTk_MexNonlinearCG::setAlgorithmParameters(dotk::DOTk_NonlinearCG & algorithm_)
{
    size_t max_num_itr = DOTk_MexAlgorithmTypeFO::getMaxNumAlgorithmItr();
    algorithm_.setMaxNumItr(max_num_itr);
    Real objective_tolerance = DOTk_MexAlgorithmTypeFO::getOptimalityTolerance();
    algorithm_.setObjectiveFuncTol(objective_tolerance);
    Real gradient_tolerance = DOTk_MexAlgorithmTypeFO::getGradientTolerance();
    algorithm_.setGradientTol(gradient_tolerance);
    Real trial_step_tolerance = DOTk_MexAlgorithmTypeFO::getTrialStepTolerance();
    algorithm_.setTrialStepTol(trial_step_tolerance);
}

void DOTk_MexNonlinearCG::solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeFO::getProblemType();
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

    dotk::DOTk_NonlinearCG algorithm(step, mng);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    dotk::types::nonlinearcg_t cg_type;
    dotk::mex::parseNonlinearCgMethod(input_[0], cg_type);
    if(cg_type == dotk::types::DANIELS_NLCG)
    {
        std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
            hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
        dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], primal, hessian);
        algorithm.setDanielsNlcg(hessian);
        algorithm.getMin();
    }
    else
    {
        this->setAlgorithmType(cg_type, algorithm);
        algorithm.getMin();
    }

    DOTk_MexAlgorithmTypeFO::gatherOutputData(algorithm, *mng, output_);
}

void DOTk_MexNonlinearCG::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeFO::getProblemType();
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

    dotk::DOTk_NonlinearCG algorithm(step, mng);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    dotk::types::nonlinearcg_t cg_type;
    dotk::mex::parseNonlinearCgMethod(input_[0], cg_type);
    if(cg_type == dotk::types::DANIELS_NLCG)
    {
        std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
            hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective, equality));
        dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], primal, hessian);
        algorithm.setDanielsNlcg(hessian);
        algorithm.getMin();
    }
    else
    {
        this->setAlgorithmType(cg_type, algorithm);
        algorithm.getMin();
    }

    DOTk_MexAlgorithmTypeFO::gatherOutputData(algorithm, *mng, output_);
}

void DOTk_MexNonlinearCG::solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr< dotk::Vector<double> > bounds = primal->control()->clone();
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

    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
        mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::mex::buildGradient(input_[0], primal, mng);

    dotk::DOTk_NonlinearCG algorithm(step, mng);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    dotk::types::nonlinearcg_t cg_type;
    dotk::mex::parseNonlinearCgMethod(input_[0], cg_type);
    if(cg_type == dotk::types::DANIELS_NLCG)
    {
        std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
            hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
        dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], primal, hessian);
        algorithm.setDanielsNlcg(hessian);
        algorithm.getMin();
    }
    else
    {
        this->setAlgorithmType(cg_type, algorithm);
        algorithm.getMin();
    }

    DOTk_MexAlgorithmTypeFO::gatherOutputData(algorithm, *mng, output_);
}

void DOTk_MexNonlinearCG::solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunctionOperators.get(), type));
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraintOperators);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraintOperators.get(), type));

    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr< dotk::Vector<double> > bounds = primal->control()->clone();
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

    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeUNP>
        mng(new dotk::DOTk_LineSearchMngTypeUNP(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], primal, mng);

    dotk::DOTk_NonlinearCG algorithm(step, mng);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    dotk::types::nonlinearcg_t cg_type;
    dotk::mex::parseNonlinearCgMethod(input_[0], cg_type);
    if(cg_type == dotk::types::DANIELS_NLCG)
    {
        std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
            hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective, equality));
        dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], primal, hessian);
        algorithm.setDanielsNlcg(hessian);
        algorithm.getMin();
    }
    else
    {
        this->setAlgorithmType(cg_type, algorithm);
        algorithm.getMin();
    }

    DOTk_MexAlgorithmTypeFO::gatherOutputData(algorithm, *mng, output_);
}

void DOTk_MexNonlinearCG::setAlgorithmType(const dotk::types::nonlinearcg_t & type_,
                                           dotk::DOTk_NonlinearCG & algorithm_)
{
    switch (type_)
    {
        case dotk::types::FLETCHER_REEVES_NLCG:
        {
            algorithm_.setFletcherReevesNlcg();
            break;
        }
        case dotk::types::POLAK_RIBIERE_NLCG:
        {
            algorithm_.setPolakRibiereNlcg();
            break;
        }
        case dotk::types::HESTENES_STIEFEL_NLCG:
        {
            algorithm_.setHestenesStiefelNlcg();
            break;
        }
        case dotk::types::CONJUGATE_DESCENT_NLCG:
        {
            algorithm_.setConjugateDescentNlcg();
            break;
        }
        case dotk::types::HAGER_ZHANG_NLCG:
        {
            algorithm_.setHagerZhangNlcg();
            break;
        }
        case dotk::types::DAI_LIAO_NLCG:
        {
            algorithm_.setDaiLiaoNlcg();
            break;
        }
        case dotk::types::DAI_YUAN_NLCG:
        {
            algorithm_.setDaiYuanNlcg();
            break;
        }
        case dotk::types::DAI_YUAN_HYBRID_NLCG:
        {
            algorithm_.setDaiYuanHybridNlcg();
            break;
        }
        case dotk::types::PERRY_SHANNO_NLCG:
        {
            algorithm_.setPerryShannoNlcg();
            break;
        }
        case dotk::types::LIU_STOREY_NLCG:
        {
            algorithm_.setLiuStoreyNlcg();
            break;
        }
        case dotk::types::DANIELS_NLCG:
        default:
        {
            std::string msg("\n DOTk/MEX WARNING: USING DEFAULT NONLINEAR CONJUGATE GRADIENT = FLETCHER REEVES. \n");
            mexWarnMsgTxt(msg.c_str());
        }
    }
}

}
