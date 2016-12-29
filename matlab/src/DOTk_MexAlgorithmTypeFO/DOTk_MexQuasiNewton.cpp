/*
 * DOTk_MexQuasiNewton.cpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>

#include "DOTk_MexVector.hpp"
#include "DOTk_MexQuasiNewton.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexQuasiNewtonParser.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeFO.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeFO.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.cpp"
#include "DOTk_MexFactoriesAlgorithmTypeGB.hpp"

#include "DOTk_Dual.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeUNP.hpp"
#include "DOTk_LineSearchQuasiNewton.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"

namespace dotk
{

DOTk_MexQuasiNewton::DOTk_MexQuasiNewton(const mxArray* options_[]) :
        dotk::DOTk_MexAlgorithmTypeFO(options_[0]),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr)
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
    dotk::types::problem_t type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, type));

    // Set line search based algorithm data manager
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
        mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::mex::buildGradient(input_[0], mng);

    this->optimize(step, mng, input_, output_);
}

void DOTk_MexQuasiNewton::solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    this->optimize(step, mng, input_, output_);
}

void DOTk_MexQuasiNewton::solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    // Set line search step manager
    dotk::types::line_search_t step_type = this->getLineSearchMethod();
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    step->build(primal, step_type);
    DOTk_MexAlgorithmTypeFO::setLineSearchStepMng(*step);
    DOTk_MexAlgorithmTypeFO::setBoundConstraintMethod(input_[0], primal, step);

    // Set objective function operators and line search based algorithm data manager
    dotk::types::problem_t type = dotk::DOTk_MexAlgorithmTypeFO::getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, type));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    dotk::mex::buildGradient(input_[0], mng);

    this->optimize(step, mng, input_, output_);
}

void DOTk_MexQuasiNewton::solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[])
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

    this->optimize(step, mng, input_, output_);
}

void DOTk_MexQuasiNewton::setAlgorithmParameters(dotk::DOTk_LineSearchQuasiNewton & algorithm_)
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

void DOTk_MexQuasiNewton::clear()
{
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
}

void DOTk_MexQuasiNewton::initialize(const mxArray* options_[])
{
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);
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
