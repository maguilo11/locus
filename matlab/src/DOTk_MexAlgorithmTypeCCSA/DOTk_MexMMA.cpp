/*
 * DOTk_MexMMA.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexMMA.hpp"
#include "DOTk_MexVector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexInequalityConstraint.hpp"

#include "DOTk_Primal.hpp"
#include "DOTk_DataMngCCSA.hpp"
#include "DOTk_AlgorithmCCSA.hpp"
#include "DOTk_SubProblemMMA.hpp"
#include "DOTk_DualSolverNLCG.hpp"

namespace dotk
{

DOTk_MexMMA::DOTk_MexMMA(const mxArray* options_[]) :
        dotk::DOTk_MexMethodCCSA(options_[0]),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr),
        m_InequalityConstraint(nullptr)
{
    this->initialize(options_);
}

DOTk_MexMMA::~DOTk_MexMMA()
{
    this->clear();
}

void DOTk_MexMMA::clear()
{
    dotk::mex::destroy(m_ObjectiveFunction);
    dotk::mex::destroy(m_EqualityConstraint);
    dotk::mex::destroy(m_InequalityConstraint);
}

void DOTk_MexMMA::solve(const mxArray* input_[], mxArray* output_[])
{
    dotk::types::problem_t type = this->getProblemType();

    switch(type)
    {
        case dotk::types::TYPE_CLP:
        {
            this->solveLinearProgrammingProblem(input_, output_);
            break;
        }
        case dotk::types::TYPE_CNLP:
        {
            this->solveNonlinearProgrammingProblem(input_, output_);
            break;
        }
        case dotk::types::TYPE_ULP:
        case dotk::types::TYPE_UNLP:
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
            std::string msg(" DOTk/MEX ERROR: Invalid Problem Type for CCSA Method. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

void DOTk_MexMMA::solveLinearProgrammingProblem(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: duals and control vectors
    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector duals(num_duals, 0.);
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
    mxDestroyArray(mx_initial_control);

    // Allocate DOTk data structures
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateUserDefinedDual(duals);
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

    // Set objective function and inequality constraint
    dotk::types::problem_t problem_type = this->getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, problem_type));
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint>
        shared_ptr(new dotk::DOTk_MexInequalityConstraint(m_InequalityConstraint, problem_type));
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality(1, shared_ptr);
    // Set data manager
    std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> data_mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    // Set dual solver
    std::tr1::shared_ptr<dotk::DOTk_DualSolverNLCG> dual_solver(new dotk::DOTk_DualSolverNLCG(primal));
    dual_solver->setNonlinearCgType(dotk::DOTk_MexMethodCCSA::getNonlinearConjugateGradientType());
    dotk::DOTk_MexMethodCCSA::setDualSolverParameters(dual_solver);

    // Set Method of Moving Asymptotes (MMA) subproblem and initialize algorithm
    std::tr1::shared_ptr<dotk::DOTk_SubProblemMMA> subproblem(new dotk::DOTk_SubProblemMMA(data_mng, dual_solver));
    dotk::DOTk_AlgorithmCCSA algorithm(data_mng, subproblem);
    dotk::DOTk_MexMethodCCSA::setPrimalSolverParameters(algorithm);
    algorithm.printDiagnosticsAtEveryItrAndSolutionAtTheEnd();

    algorithm.getMin();

    dotk::DOTk_MexMethodCCSA::gatherOutputData(algorithm, data_mng, output_);
}

void DOTk_MexMMA::solveNonlinearProgrammingProblem(const mxArray* input_[], mxArray* output_[])
{
    // Set core data structures: duals, states, and control vectors
    size_t num_duals = dotk::mex::parseNumberDuals(input_[0]);
    dotk::MexVector duals(num_duals, 0.);
    size_t num_states = dotk::mex::parseNumberStates(input_[0]);
    dotk::MexVector states(num_states, 0.);
    mxArray* mx_initial_control = dotk::mex::parseInitialControl(input_[0]);
    dotk::MexVector controls(mx_initial_control);
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

    // Set objective function, equality constraint, and inequality constraint
    dotk::types::problem_t type = this->getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction>
        objective(new dotk::DOTk_MexObjectiveFunction(m_ObjectiveFunction, type));
    m_EqualityConstraint = dotk::mex::parseEqualityConstraint(input_[1]);
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint>
        equality(new dotk::DOTk_MexEqualityConstraint(m_EqualityConstraint, type));
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint>
        inequality(new dotk::DOTk_MexInequalityConstraint(m_InequalityConstraint, type));
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality_vector(1, inequality);

    // Set data and assembly operators manager
    std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA>
        data_mng(new dotk::DOTk_DataMngCCSA(primal, objective, equality, inequality_vector));

    // Set dual solver
    std::tr1::shared_ptr<dotk::DOTk_DualSolverNLCG> dual_solver(new dotk::DOTk_DualSolverNLCG(primal));
    dual_solver->setNonlinearCgType(dotk::DOTk_MexMethodCCSA::getNonlinearConjugateGradientType());
    dotk::DOTk_MexMethodCCSA::setDualSolverParameters(dual_solver);

    // Set MMA subproblem and initialize algorithm
    std::tr1::shared_ptr<dotk::DOTk_SubProblemMMA> subproblem(new dotk::DOTk_SubProblemMMA(data_mng, dual_solver));
    dotk::DOTk_AlgorithmCCSA algorithm(data_mng, subproblem);
    dotk::DOTk_MexMethodCCSA::setPrimalSolverParameters(algorithm);
    algorithm.printDiagnosticsAtEveryItrAndSolutionAtTheEnd();

    algorithm.getMin();

    dotk::DOTk_MexMethodCCSA::gatherOutputData(algorithm, data_mng, output_);
}

void DOTk_MexMMA::initialize(const mxArray* options_[])
{
    m_ObjectiveFunction = dotk::mex::parseObjectiveFunction(options_[1]);
    m_InequalityConstraint = dotk::mex::parseInequalityConstraint(options_[1]);
}

}
