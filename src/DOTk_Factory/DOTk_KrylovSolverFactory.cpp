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

DOTk_KrylovSolverFactory::DOTk_KrylovSolverFactory(dotk::types::krylov_solver_t aType) :
        mWarningMsg(),
        mFactoryType(aType)
{
}

DOTk_KrylovSolverFactory::~DOTk_KrylovSolverFactory()
{
}

void DOTk_KrylovSolverFactory::setWarningMsg(const std::string & aMsg)
{
    mWarningMsg.append(aMsg);
}
std::string DOTk_KrylovSolverFactory::getWarningMsg() const
{
    return (mWarningMsg);
}

dotk::types::krylov_solver_t DOTk_KrylovSolverFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_KrylovSolverFactory::setFactoryType(dotk::types::krylov_solver_t aType)
{
    mFactoryType = aType;
}

void DOTk_KrylovSolverFactory::buildPrecGmresSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                    const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                    size_t aMaxNumIterations,
                                                    std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_PrecGMRES>(aPrimal, aLinearOperator, aMaxNumIterations);
    this->setFactoryType(dotk::types::PREC_GMRES);
}
void DOTk_KrylovSolverFactory::buildLeftPrecCgSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                     const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                     std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_LeftPrecCG>(aPrimal, aLinearOperator);
    this->setFactoryType(dotk::types::LEFT_PREC_CG);
}

void DOTk_KrylovSolverFactory::buildLeftPrecCrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                     const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                     std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_LeftPrecCR>(aPrimal, aLinearOperator);
    this->setFactoryType(dotk::types::LEFT_PREC_CGNE);
}

void DOTk_KrylovSolverFactory::buildLeftPrecGcrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                      const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                      size_t aMaxNumIterations,
                                                      std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_LeftPrecGCR>(aPrimal, aLinearOperator, aMaxNumIterations);
    this->setFactoryType(dotk::types::LEFT_PREC_CGNE);
}

void DOTk_KrylovSolverFactory::buildLeftPrecCgneSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                       const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                       std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_LeftPrecCGNE>(aPrimal, aLinearOperator);
    this->setFactoryType(dotk::types::LEFT_PREC_CGNE);
}

void DOTk_KrylovSolverFactory::buildLeftPrecCgnrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                       const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                       std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_LeftPrecCGNR>(aPrimal, aLinearOperator);
    this->setFactoryType(dotk::types::LEFT_PREC_CGNE);
}

void DOTk_KrylovSolverFactory::buildProjLeftPrecCgSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                                         const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                                         size_t aMaxNumIterations,
                                                         std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_ProjectedLeftPrecCG>(aPrimal, aLinearOperator, aMaxNumIterations);
    this->setFactoryType(dotk::types::PROJECTED_PREC_CG);
}

void DOTk_KrylovSolverFactory::build(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aMng,
                                     std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput)
{
    switch(this->getFactoryType())
    {
        case dotk::types::LEFT_PREC_CG:
        {
            aOutput = std::make_shared<dotk::DOTk_LeftPrecCG>(aMng);
            break;
        }
        case dotk::types::PREC_GMRES:
        {
            aOutput = std::make_shared<dotk::DOTk_PrecGMRES>(aMng);
            break;
        }
        case dotk::types::LEFT_PREC_CGNR:
        {
            aOutput = std::make_shared<dotk::DOTk_LeftPrecCGNR>(aMng);
            break;
        }
        case dotk::types::LEFT_PREC_CGNE:
        {
            aOutput = std::make_shared<dotk::DOTk_LeftPrecCGNE>(aMng);
            break;
        }
        case dotk::types::LEFT_PREC_CR:
        {
            aOutput = std::make_shared<dotk::DOTk_LeftPrecCR>(aMng);
            break;
        }
        case dotk::types::LEFT_PREC_GCR:
        {
            aOutput = std::make_shared<dotk::DOTk_LeftPrecGCR>(aMng);
            break;
        }
        case dotk::types::PROJECTED_PREC_CG:
        {
            aOutput = std::make_shared<dotk::DOTk_ProjectedLeftPrecCG>(aMng);
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
            aOutput = std::make_shared<dotk::DOTk_LeftPrecCG>(aMng);
            break;
        }
    }
}

}
