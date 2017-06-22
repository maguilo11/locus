/*
 * DOTk_AugmentedSystemPrecFactory.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Types.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_AugmentedSystemLeftPrec.hpp"
#include "DOTk_AugmentedSystemPrecFactory.hpp"

namespace dotk
{

DOTk_AugmentedSystemPrecFactory::DOTk_AugmentedSystemPrecFactory(size_t max_num_solver_itr_) :
        m_MaxNumSolverItr(max_num_solver_itr_)
{
}

DOTk_AugmentedSystemPrecFactory::~DOTk_AugmentedSystemPrecFactory()
{
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithPcgSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & prec_)
{
    prec_.reset(new dotk::DOTk_AugmentedSystemLeftPrec(primal_));
    prec_->setLeftPrecCgSolver(primal_, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithGcrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & prec_)
{
    prec_.reset(new dotk::DOTk_AugmentedSystemLeftPrec(primal_));
    prec_->setLeftPrecGcrSolver(primal_, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithCrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & prec_)
{
    prec_.reset(new dotk::DOTk_AugmentedSystemLeftPrec(primal_));
    prec_->setLeftPrecCrSolver(primal_, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithCgneSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & prec_)
{
    prec_.reset(new dotk::DOTk_AugmentedSystemLeftPrec(primal_));
    prec_->setLeftPrecCgneSolver(primal_, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithCgnrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & prec_)
{
    prec_.reset(new dotk::DOTk_AugmentedSystemLeftPrec(primal_));
    prec_->setLeftPrecCgnrSolver(primal_, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithGmresSolver
(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & prec_)
{
    prec_.reset(new dotk::DOTk_AugmentedSystemLeftPrec(primal_));
    prec_->setPrecGmresSolver(primal_, m_MaxNumSolverItr);
}

}
