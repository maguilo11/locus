/*
 * DOTk_MexGCMMA.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */


#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MexGCMMA.hpp"
#include "DOTk_DataMngCCSA.hpp"
#include "DOTk_AlgorithmCCSA.hpp"
#include "DOTk_DualSolverNLCG.hpp"
#include "DOTk_SubProblemGCMMA.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexMethodCcsaParser.hpp"
#include "DOTk_MexContainerFactory.hpp"
#include "DOTk_MexObjectiveFunction.cpp"
#include "DOTk_MexObjectiveFunction.hpp"
#include "DOTk_MexEqualityConstraint.cpp"
#include "DOTk_MexEqualityConstraint.hpp"
#include "DOTk_MexInequalityConstraint.cpp"
#include "DOTk_MexInequalityConstraint.hpp"

namespace dotk
{

DOTk_MexGCMMA::DOTk_MexGCMMA(const mxArray* options_[]) :
        dotk::DOTk_MexMethodCCSA(options_[0]),
        m_MaxNumberSubProblemIterations(10),
        m_SubProblemResidualTolerance(1e-6),
        m_SubProblemStagnationTolerance(1e-6),
        m_ObjectiveFunction(nullptr),
        m_EqualityConstraint(nullptr),
        m_InequalityConstraint(nullptr)
{
    this->initialize(options_);
}

DOTk_MexGCMMA::~DOTk_MexGCMMA()
{
    this->clear();
}

void DOTk_MexGCMMA::clear()
{
    m_ObjectiveFunction.release();
    m_EqualityConstraint.release();
    m_InequalityConstraint.release();
}

void DOTk_MexGCMMA::solve(const mxArray* input_[], mxArray* output_[])
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

void DOTk_MexGCMMA::solveLinearProgrammingProblem(const mxArray* input_[], mxArray* output_[])
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

    std::tr1::shared_ptr<dotk::DOTk_SubProblemGCMMA> sub_problem(new dotk::DOTk_SubProblemGCMMA(data_mng, dual_solver));
    sub_problem->setMaxNumIterations(m_MaxNumberSubProblemIterations);
    sub_problem->setResidualTolerance(m_SubProblemResidualTolerance);
    sub_problem->setStagnationTolerance(m_SubProblemStagnationTolerance);

    dotk::DOTk_AlgorithmCCSA algorithm(data_mng, sub_problem);
    dotk::DOTk_MexMethodCCSA::setPrimalSolverParameters(algorithm);

    algorithm.printDiagnosticsAtEveryItrAndSolutionAtTheEnd();
    algorithm.getMin();

    dotk::DOTk_MexMethodCCSA::gatherOutputData(algorithm, data_mng, output_);
}

void DOTk_MexGCMMA::solveNonlinearProgrammingProblem(const mxArray* input_[], mxArray* output_[])
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

    std::tr1::shared_ptr<dotk::DOTk_SubProblemGCMMA> sub_problem(new dotk::DOTk_SubProblemGCMMA(data_mng, dual_solver));
    sub_problem->setMaxNumIterations(m_MaxNumberSubProblemIterations);
    sub_problem->setResidualTolerance(m_SubProblemResidualTolerance);
    sub_problem->setStagnationTolerance(m_SubProblemStagnationTolerance);

    dotk::DOTk_AlgorithmCCSA algorithm(data_mng, sub_problem);
    dotk::DOTk_MexMethodCCSA::setPrimalSolverParameters(algorithm);

    algorithm.printDiagnosticsAtEveryItrAndSolutionAtTheEnd();
    algorithm.getMin();

    dotk::DOTk_MexMethodCCSA::gatherOutputData(algorithm, data_mng, output_);
}

void DOTk_MexGCMMA::initialize(const mxArray* options_[])
{
    dotk::mex::parseObjectiveFunction(options_[1], m_ObjectiveFunction);
    dotk::mex::parseInequalityConstraint(options_[1], m_InequalityConstraint);

    dotk::mex::parseMaxNumberSubProblemIterations(options_[0], m_MaxNumberSubProblemIterations);
    dotk::mex::parseSubProblemResidualTolerance(options_[0], m_SubProblemResidualTolerance);
    dotk::mex::parseSubProblemStagnationTolerance(options_[0], m_SubProblemStagnationTolerance);
}

}
