/*
 * DOTk_MexMMA.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */


#include "vector.hpp"
#include "DOTk_MexMMA.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_DataMngCCSA.hpp"
#include "DOTk_AlgorithmCCSA.hpp"
#include "DOTk_SubProblemMMA.hpp"
#include "DOTk_DualSolverNLCG.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexInequalityConstraint.cpp"
#include "DOTk_MexInequalityConstraint.hpp"

namespace dotk
{

DOTk_MexMMA::DOTk_MexMMA(const mxArray* options_[]) :
        dotk::DOTk_MexMethodCCSA(options_[0]),
        m_ObjectiveFunction(NULL),
        m_EqualityConstraint(NULL),
        m_InequalityConstraint(NULL)
{
    this->initialize(options_);
}

DOTk_MexMMA::~DOTk_MexMMA()
{
    this->clear();
}

void DOTk_MexMMA::clear()
{
    m_ObjectiveFunction.release();
    m_EqualityConstraint.release();
    m_InequalityConstraint.release();
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
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildControlContainer(input_[0], *primal);

    std::tr1::shared_ptr< dotk::Vector<double> > vector = primal->control()->clone();
    dotk::mex::parseControlLowerBound(input_[0], *vector);
    primal->setControlLowerBound(*vector);
    vector->fill(0.);
    dotk::mex::parseControlUpperBound(input_[0], *vector);
    primal->setControlUpperBound(*vector);

    dotk::types::problem_t type = this->getProblemType();
    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunction.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint<double> >
        shared_ptr(new dotk::DOTk_MexInequalityConstraint<double>(m_InequalityConstraint.get(), type));
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality(1, shared_ptr);

    std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> data_mng(new dotk::DOTk_DataMngCCSA(primal, objective, inequality));

    std::tr1::shared_ptr<dotk::DOTk_DualSolverNLCG> dual_solver(new dotk::DOTk_DualSolverNLCG(primal));
    dual_solver->setNonlinearCgType(dotk::DOTk_MexMethodCCSA::getNonlinearConjugateGradientType());
    dotk::DOTk_MexMethodCCSA::setDualSolverParameters(dual_solver);

    std::tr1::shared_ptr<dotk::DOTk_SubProblemMMA> sub_problem(new dotk::DOTk_SubProblemMMA(data_mng, dual_solver));
    dotk::DOTk_AlgorithmCCSA algorithm(data_mng, sub_problem);
    dotk::DOTk_MexMethodCCSA::setPrimalSolverParameters(algorithm);

    algorithm.printDiagnosticsAtEveryItrAndSolutionAtTheEnd();
    algorithm.getMin();

    dotk::DOTk_MexMethodCCSA::gatherOutputData(algorithm, data_mng, output_);
}

void DOTk_MexMMA::solveNonlinearProgrammingProblem(const mxArray* input_[], mxArray* output_[])
{
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    dotk::mex::buildDualContainer(input_[0], *primal);
    dotk::mex::buildStateContainer(input_[0], *primal);
    dotk::mex::buildPrimalContainer(input_[0], *primal);

    std::tr1::shared_ptr< dotk::Vector<double> > vector = primal->control()->clone();
    dotk::mex::parseControlLowerBound(input_[0], *vector);
    primal->setControlLowerBound(*vector);
    vector->fill(0.);
    dotk::mex::parseControlUpperBound(input_[0], *vector);
    primal->setControlUpperBound(*vector);

    dotk::types::problem_t type = this->getProblemType();
    dotk::mex::parseEqualityConstraint(input_[1], m_EqualityConstraint);

    std::tr1::shared_ptr<dotk::DOTk_MexObjectiveFunction<double> >
        objective(new dotk::DOTk_MexObjectiveFunction<double>(m_ObjectiveFunction.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexEqualityConstraint<double> >
        equality(new dotk::DOTk_MexEqualityConstraint<double>(m_EqualityConstraint.get(), type));
    std::tr1::shared_ptr<dotk::DOTk_MexInequalityConstraint<double> >
        shared_ptr(new dotk::DOTk_MexInequalityConstraint<double>(m_InequalityConstraint.get(), type));
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<double> > > inequality(1, shared_ptr);

    std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> data_mng(new dotk::DOTk_DataMngCCSA(primal, objective, equality, inequality));

    std::tr1::shared_ptr<dotk::DOTk_DualSolverNLCG> dual_solver(new dotk::DOTk_DualSolverNLCG(primal));
    dual_solver->setNonlinearCgType(dotk::DOTk_MexMethodCCSA::getNonlinearConjugateGradientType());
    dotk::DOTk_MexMethodCCSA::setDualSolverParameters(dual_solver);

    std::tr1::shared_ptr<dotk::DOTk_SubProblemMMA> sub_problem(new dotk::DOTk_SubProblemMMA(data_mng, dual_solver));
    dotk::DOTk_AlgorithmCCSA algorithm(data_mng, sub_problem);
    dotk::DOTk_MexMethodCCSA::setPrimalSolverParameters(algorithm);

    algorithm.printDiagnosticsAtEveryItrAndSolutionAtTheEnd();
    algorithm.getMin();

    dotk::DOTk_MexMethodCCSA::gatherOutputData(algorithm, data_mng, output_);
}

void DOTk_MexMMA::initialize(const mxArray* options_[])
{
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunction);
    dotk::mex::parseInequalityConstraint(options_[1], m_InequalityConstraint);
}

}
