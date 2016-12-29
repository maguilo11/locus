/*
 * DOTk_MexNonlinearCG.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>
#include <iostream>

#include "DOTk_MexVector.hpp"
#include "DOTk_MexNonlinearCG.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexNumDiffHessianFactory.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

#include "DOTk_Primal.hpp"
#include "DOTk_NonlinearCG.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"

namespace dotk
{

DOTk_MexNonlinearCG::DOTk_MexNonlinearCG(const mxArray* options_[]) :
        dotk::DOTk_MexAlgorithmTypeFO(options_[0]),
        m_NonlinearCgType(dotk::types::UNDEFINED_NLCG),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr)
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
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
}

void DOTk_MexNonlinearCG::initialize(const mxArray* options_[])
{
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);
}

void DOTk_MexNonlinearCG::setAlgorithmParameters(dotk::DOTk_NonlinearCG & algorithm_)
{
    size_t max_num_itr = DOTk_MexAlgorithmTypeFO::getMaxNumAlgorithmItr();
    algorithm_.setMaxNumItr(max_num_itr);
    Real objective_tolerance = DOTk_MexAlgorithmTypeFO::getObjectiveTolerance();
    algorithm_.setObjectiveFuncTol(objective_tolerance);
    Real gradient_tolerance = DOTk_MexAlgorithmTypeFO::getGradientTolerance();
    algorithm_.setGradientTol(gradient_tolerance);
    Real trial_step_tolerance = DOTk_MexAlgorithmTypeFO::getTrialStepTolerance();
    algorithm_.setTrialStepTol(trial_step_tolerance);
}

void DOTk_MexNonlinearCG::solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: control vector
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateUserDefinedControl(controls);

    // Set line search step manager
    dotk::types::line_search_t step_type = this->getLineSearchMethod();
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    step->build(primal, step_type);
    DOTk_MexAlgorithmTypeFO::setLineSearchStepMng(*step);

    // Set objective function operators
    dotk::types::problem_t problem_type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, problem_type));

    // Set line search based algorithm data manager
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
        mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::mex::buildGradient(input_[0], mng);

    // Initialize nonlinear conjugate gradient algorithm
    dotk::DOTk_NonlinearCG algorithm(step, mng);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    // Solve optimization problem
    dotk::types::nonlinearcg_t nlcg_type = dotk::mex::parseNonlinearCgMethod(input_[0]);
    if(nlcg_type == dotk::types::DANIELS_NLCG)
    {
        std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
            hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
        dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);
        algorithm.setDanielsNlcg(hessian);
        algorithm.getMin();
    }
    else
    {
        this->setAlgorithmType(nlcg_type, algorithm);
        algorithm.getMin();
    }

    DOTk_MexAlgorithmTypeFO::gatherOutputData(algorithm, *mng, output_);
}

void DOTk_MexNonlinearCG::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    // Set objective function and equality constraint operators
    dotk::types::problem_t problem_type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, problem_type));
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality(new dotk::DOTk_MexEqualityConstraint(m_EqualityConstraint, problem_type));

    // Set line search step manager
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    dotk::types::line_search_t step_type = this->getLineSearchMethod();
    step->build(primal, step_type);
    DOTk_MexAlgorithmTypeFO::setLineSearchStepMng(*step);

    // Set line search based algorithm manager
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeUNP>
        mng(new dotk::DOTk_LineSearchMngTypeUNP(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], mng);

    // Initialize nonlinear conjugate gradient algorithm
    dotk::DOTk_NonlinearCG algorithm(step, mng);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    // Solve optimization problem
    dotk::types::nonlinearcg_t nlcg_type = dotk::mex::parseNonlinearCgMethod(input_[0]);
    if(nlcg_type == dotk::types::DANIELS_NLCG)
    {
        std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
            hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective, equality));
        dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);
        algorithm.setDanielsNlcg(hessian);
        algorithm.getMin();
    }
    else
    {
        this->setAlgorithmType(nlcg_type, algorithm);
        algorithm.getMin();
    }
    DOTk_MexAlgorithmTypeFO::gatherOutputData(algorithm, *mng, output_);
}

void DOTk_MexNonlinearCG::solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: control vectors
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);
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

    // Set line search step manager
    dotk::types::line_search_t step_type = this->getLineSearchMethod();
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    step->build(primal, step_type);
    DOTk_MexAlgorithmTypeFO::setLineSearchStepMng(*step);
    DOTk_MexAlgorithmTypeFO::setBoundConstraintMethod(input_[0], primal, step);

    // Set objective function operators and line search based algorithm data manager
    dotk::types::problem_t problem_type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, problem_type));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::mex::buildGradient(input_[0], mng);

    // Initialize nonlinear conjugate gradient algorithm
    dotk::DOTk_NonlinearCG algorithm(step, mng);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    // Solve optimization problem
    dotk::types::nonlinearcg_t nlcg_type = dotk::mex::parseNonlinearCgMethod(input_[0]);
    if(nlcg_type == dotk::types::DANIELS_NLCG)
    {
        std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
            hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
        dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);
        algorithm.setDanielsNlcg(hessian);
        algorithm.getMin();
    }
    else
    {
        this->setAlgorithmType(nlcg_type, algorithm);
        algorithm.getMin();
    }
    DOTk_MexAlgorithmTypeFO::gatherOutputData(algorithm, *mng, output_);
}

void DOTk_MexNonlinearCG::solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    dotk::types::line_search_t step_type = this->getLineSearchMethod();
    step->build(primal, step_type);
    DOTk_MexAlgorithmTypeFO::setLineSearchStepMng(*step);
    DOTk_MexAlgorithmTypeFO::setBoundConstraintMethod(input_[0], primal, step);

    // Set objective function and equality constraint operators
    dotk::types::problem_t type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, type));
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality(new dotk::DOTk_MexEqualityConstraint(m_EqualityConstraint, type));

    // Set line search based optimization algorithm data manager
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeUNP>
        mng(new dotk::DOTk_LineSearchMngTypeUNP(primal, objective, equality));
    dotk::mex::buildGradient(input_[0], mng);

    // Initialize nonlinear conjugate gradient algorithm
    dotk::DOTk_NonlinearCG algorithm(step, mng);
    this->setAlgorithmParameters(algorithm);
    algorithm.printDiagnosticsEveryItrAndSolutionAtTheEnd();

    // Solve optimization problem
    dotk::types::nonlinearcg_t nlcg_type = dotk::mex::parseNonlinearCgMethod(input_[0]);
    if(nlcg_type == dotk::types::DANIELS_NLCG)
    {
        std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian>
            hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective, equality));
        dotk::mex::buildNumericallyDifferentiatedHessian(input_[0], controls, hessian);
        algorithm.setDanielsNlcg(hessian);
        algorithm.getMin();
    }
    else
    {
        this->setAlgorithmType(nlcg_type, algorithm);
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
            std::string msg("\n DOTk WARNING: USING DEFAULT NONLINEAR CONJUGATE GRADIENT = FLETCHER REEVES. \n");
            mexWarnMsgTxt(msg.c_str());
        }
    }
}

}
