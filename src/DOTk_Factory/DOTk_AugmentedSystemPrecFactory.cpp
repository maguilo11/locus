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

DOTk_AugmentedSystemPrecFactory::DOTk_AugmentedSystemPrecFactory(size_t aMaxNumSolverIterations) :
        m_MaxNumSolverItr(aMaxNumSolverIterations)
{
}

DOTk_AugmentedSystemPrecFactory::~DOTk_AugmentedSystemPrecFactory()
{
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithPcgSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner)
{
    aPreconditioner = std::make_shared<dotk::DOTk_AugmentedSystemLeftPrec>(aPrimal);
    aPreconditioner->setLeftPrecCgSolver(aPrimal, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithGcrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner)
{
    aPreconditioner = std::make_shared<dotk::DOTk_AugmentedSystemLeftPrec>(aPrimal);
    aPreconditioner->setLeftPrecGcrSolver(aPrimal, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithCrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner)
{
    aPreconditioner = std::make_shared<dotk::DOTk_AugmentedSystemLeftPrec>(aPrimal);
    aPreconditioner->setLeftPrecCrSolver(aPrimal, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithCgneSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner)
{
    aPreconditioner = std::make_shared<dotk::DOTk_AugmentedSystemLeftPrec>(aPrimal);
    aPreconditioner->setLeftPrecCgneSolver(aPrimal, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithCgnrSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner)
{
    aPreconditioner = std::make_shared<dotk::DOTk_AugmentedSystemLeftPrec>(aPrimal);
    aPreconditioner->setLeftPrecCgnrSolver(aPrimal, m_MaxNumSolverItr);
}

void DOTk_AugmentedSystemPrecFactory::buildAugmentedSystemPrecWithGmresSolver
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner)
{
    aPreconditioner = std::make_shared<dotk::DOTk_AugmentedSystemLeftPrec>(aPrimal);
    aPreconditioner->setPrecGmresSolver(aPrimal, m_MaxNumSolverItr);
}

}
