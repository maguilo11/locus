/*
 * DOTk_DirectSolverFactory.cpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "DOTk_DirectSolver.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_DirectSolverFactory.hpp"
#include "DOTk_LowerTriangularDirectSolver.hpp"
#include "DOTk_UpperTriangularDirectSolver.hpp"

namespace dotk
{

DOTk_DirectSolverFactory::DOTk_DirectSolverFactory() :
        mErrorMsg(),
        mFactoryType(dotk::types::DIRECT_SOLVER_DISABLED)
{
}

DOTk_DirectSolverFactory::DOTk_DirectSolverFactory(dotk::types::direct_solver_t type_) :
        mErrorMsg(),
        mFactoryType(type_)
{
}

DOTk_DirectSolverFactory::~DOTk_DirectSolverFactory()
{
}

void DOTk_DirectSolverFactory::setErrorMsg(const std::string & msg_)
{
    mErrorMsg.append(msg_);
}

std::string DOTk_DirectSolverFactory::getWarningMsg() const
{
    return (mErrorMsg);
}

void DOTk_DirectSolverFactory::setFactoryType(dotk::types::direct_solver_t type_)
{
    mFactoryType = type_;
}

dotk::types::direct_solver_t DOTk_DirectSolverFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_DirectSolverFactory::buildLowerTriangularDirectSolver(std::tr1::shared_ptr<dotk::DOTk_DirectSolver> & direct_solver_)
{
    direct_solver_.reset(new dotk::DOTk_LowerTriangularDirectSolver());
    this->setFactoryType(dotk::types::LOWER_TRIANGULAR_DIRECT_SOLVER);
}

void DOTk_DirectSolverFactory::buildUpperTriangularDirectSolver(std::tr1::shared_ptr<dotk::DOTk_DirectSolver> & direct_solver_)
{
    direct_solver_.reset(new dotk::DOTk_UpperTriangularDirectSolver());
    this->setFactoryType(dotk::types::UPPER_TRIANGULAR_DIRECT_SOLVER);
}

void DOTk_DirectSolverFactory::build(std::tr1::shared_ptr<dotk::DOTk_DirectSolver> & direct_solver_)
{
    switch(this->getFactoryType())
    {
        case dotk::types::LOWER_TRIANGULAR_DIRECT_SOLVER:
        {
            direct_solver_.reset(new dotk::DOTk_LowerTriangularDirectSolver());
            break;
        }
        case dotk::types::UPPER_TRIANGULAR_DIRECT_SOLVER:
        {
            direct_solver_.reset(new dotk::DOTk_UpperTriangularDirectSolver());
            break;
        }
        case dotk::types::DIRECT_SOLVER_DISABLED:
        case dotk::types::USER_DEFINED_DIRECT_SOLVER:
        {
            break;
        }
        default:
        {
            std::ostringstream msg;
            msg << "\nDOTk ERROR: Invalid direct solver type. Please select one of the direct "
                << "solver options described in the Users' Manual.\n" << std::flush;
            dotk::ioUtils::printMessage(msg);
            this->setErrorMsg(msg.str());
            break;
        }
    }
}

}
