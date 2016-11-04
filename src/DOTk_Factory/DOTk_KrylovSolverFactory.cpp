/*
 * DOTk_KrylovSolverFactory.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "DOTk_Primal.hpp"
#include "DOTk_PrecGMRES.hpp"
#include "DOTk_LeftPrecCG.hpp"
#include "DOTk_LeftPrecCR.hpp"
#include "DOTk_LeftPrecGCR.hpp"
#include "DOTk_LeftPrecCGNR.hpp"
#include "DOTk_LeftPrecCGNE.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_ProjectedLeftPrecCG.hpp"
#include "DOTk_KrylovSolverFactory.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

DOTk_KrylovSolverFactory::DOTk_KrylovSolverFactory() :
        mWarningMsg(),
        mFactoryType(dotk::types::KRYLOV_SOLVER_DISABLED)
{
}

DOTk_KrylovSolverFactory::DOTk_KrylovSolverFactory(dotk::types::krylov_solver_t type_) :
        mWarningMsg(),
        mFactoryType(type_)
{
}

DOTk_KrylovSolverFactory::~DOTk_KrylovSolverFactory()
{
}

void DOTk_KrylovSolverFactory::setWarningMsg(const std::string & msg_)
{
    mWarningMsg.append(msg_);
}
std::string DOTk_KrylovSolverFactory::getWarningMsg() const
{
    return (mWarningMsg);
}

dotk::types::krylov_solver_t DOTk_KrylovSolverFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_KrylovSolverFactory::setFactoryType(dotk::types::krylov_solver_t type_)
{
    mFactoryType = type_;
}

void DOTk_KrylovSolverFactory::buildPrecGmresSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                    const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                    size_t max_num_itr_,
                                                    std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_)
{
    solver_.reset(new dotk::DOTk_PrecGMRES(primal_, linear_operator_, max_num_itr_));
    this->setFactoryType(dotk::types::PREC_GMRES);
}
void DOTk_KrylovSolverFactory::buildLeftPrecCgSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                     std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_)
{
    solver_.reset(new dotk::DOTk_LeftPrecCG(primal_, linear_operator_));
    this->setFactoryType(dotk::types::LEFT_PREC_CG);
}

void DOTk_KrylovSolverFactory::buildLeftPrecCrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                     const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                     std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_)
{
    solver_.reset(new dotk::DOTk_LeftPrecCR(primal_, linear_operator_));
    this->setFactoryType(dotk::types::LEFT_PREC_CGNE);
}

void DOTk_KrylovSolverFactory::buildLeftPrecGcrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                      const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                      size_t max_num_itr_,
                                                      std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_)
{
    solver_.reset(new dotk::DOTk_LeftPrecGCR(primal_, linear_operator_, max_num_itr_));
    this->setFactoryType(dotk::types::LEFT_PREC_CGNE);
}

void DOTk_KrylovSolverFactory::buildLeftPrecCgneSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                       std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_)
{
    solver_.reset(new dotk::DOTk_LeftPrecCGNE(primal_, linear_operator_));
    this->setFactoryType(dotk::types::LEFT_PREC_CGNE);
}

void DOTk_KrylovSolverFactory::buildLeftPrecCgnrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                       std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_)
{
    solver_.reset(new dotk::DOTk_LeftPrecCGNR(primal_, linear_operator_));
    this->setFactoryType(dotk::types::LEFT_PREC_CGNE);
}

void DOTk_KrylovSolverFactory::buildProjLeftPrecCgSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                         const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                                         size_t max_num_itr_,
                                                         std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_)
{
    solver_.reset(new dotk::DOTk_ProjectedLeftPrecCG(primal_, linear_operator_, max_num_itr_));
    this->setFactoryType(dotk::types::PROJECTED_PREC_CG);
}

void DOTk_KrylovSolverFactory::build(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_,
                                     std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> & solver_)
{
    switch(this->getFactoryType())
    {
        case dotk::types::LEFT_PREC_CG:
        {
            solver_.reset(new dotk::DOTk_LeftPrecCG(mng_));
            break;
        }
        case dotk::types::PREC_GMRES:
        {
            solver_.reset(new dotk::DOTk_PrecGMRES(mng_));
            break;
        }
        case dotk::types::LEFT_PREC_CGNR:
        {
            solver_.reset(new dotk::DOTk_LeftPrecCGNR(mng_));
            break;
        }
        case dotk::types::LEFT_PREC_CGNE:
        {
            solver_.reset(new dotk::DOTk_LeftPrecCGNE(mng_));
            break;
        }
        case dotk::types::LEFT_PREC_CR:
        {
            solver_.reset(new dotk::DOTk_LeftPrecCR(mng_));
            break;
        }
        case dotk::types::LEFT_PREC_GCR:
        {
            solver_.reset(new dotk::DOTk_LeftPrecGCR(mng_));
            break;
        }
        case dotk::types::PROJECTED_PREC_CG:
        {
            solver_.reset(new dotk::DOTk_ProjectedLeftPrecCG(mng_));
            break;
        }
        case dotk::types::LANCZOS:
        case dotk::types::BICG:
        case dotk::types::BICG_STAB:
        case dotk::types::USER_DEFINED_KRYLOV_SOLVER:
        case dotk::types::KRYLOV_SOLVER_DISABLED:
        {
            break;
        }
        default:
        {
            std::ostringstream msg;
            msg << "\nDOTk WARNING: Invalid Krylov solver type, Default Krylov solver was set to Precondtioned "
                    << "Conjugate Gradient (PrecCG) with IDENTITY left preconditioner.\n" << std::flush;
            dotk::ioUtils::printMessage(msg);
            this->setWarningMsg(msg.str());
            solver_.reset(new dotk::DOTk_LeftPrecCG(mng_));
            break;
        }
    }
}

}
