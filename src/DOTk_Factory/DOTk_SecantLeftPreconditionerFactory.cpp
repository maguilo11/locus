/*
 * DOTk_SecantLeftPreconditionerFactory.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Types.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTK_SecantLeftPreconditioner.hpp"
#include "DOTk_SecantLeftPreconditionerFactory.hpp"

namespace dotk
{

DOTk_SecantLeftPreconditionerFactory::DOTk_SecantLeftPreconditionerFactory() :
        m_SecantType(dotk::types::INV_HESS_DISABLED),
        m_FactoryType(dotk::types::SECANT_LEFT_PRECONDITIONER)
{
}

DOTk_SecantLeftPreconditionerFactory::~DOTk_SecantLeftPreconditionerFactory()
{
}

dotk::types::invhessian_t DOTk_SecantLeftPreconditionerFactory::getSecantType() const
{
    return (m_SecantType);
}

void DOTk_SecantLeftPreconditionerFactory::buildLdfpSecantPreconditioner
(size_t aSecantStorageSize,
 const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aOutput)
{
    dotk::types::invhessian_t tType = dotk::types::LDFP_INV_HESS;
    aOutput = std::make_shared<dotk::DOTK_SecantLeftPreconditioner>(aVector, tType, aSecantStorageSize);
    this->setSecantType(tType);
}

void DOTk_SecantLeftPreconditionerFactory::buildLsr1SecantPreconditioner
(size_t aSecantStorageSize,
 const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aOutput)
{
    dotk::types::invhessian_t tType = dotk::types::LSR1_INV_HESS;
    aOutput = std::make_shared<dotk::DOTK_SecantLeftPreconditioner>(aVector, tType, aSecantStorageSize);
    this->setSecantType(tType);
}

void DOTk_SecantLeftPreconditionerFactory::buildLbfgsSecantPreconditioner
(size_t aSecantStorageSize,
 const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aOutput)
{
    dotk::types::invhessian_t tType = dotk::types::LBFGS_INV_HESS;
    aOutput = std::make_shared<dotk::DOTK_SecantLeftPreconditioner>(aVector, tType, aSecantStorageSize);
    this->setSecantType(tType);
}

void DOTk_SecantLeftPreconditionerFactory::buildBfgsSecantPreconditioner
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aOutput)
{
    dotk::types::invhessian_t tType = dotk::types::BFGS_INV_HESS;
    aOutput = std::make_shared<dotk::DOTK_SecantLeftPreconditioner>(aVector, tType);
    this->setSecantType(tType);
}

void DOTk_SecantLeftPreconditionerFactory::buildSr1SecantPreconditioner
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aOutput)
{
    dotk::types::invhessian_t tType = dotk::types::SR1_INV_HESS;
    aOutput = std::make_shared<dotk::DOTK_SecantLeftPreconditioner>(aVector, tType);
    this->setSecantType(tType);
}

void DOTk_SecantLeftPreconditionerFactory::buildBarzilaiBorweinSecantPreconditioner
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aOutput)
{
    dotk::types::invhessian_t tType = dotk::types::BARZILAIBORWEIN_INV_HESS;
    aOutput = std::make_shared<dotk::DOTK_SecantLeftPreconditioner>(aVector, tType);
    this->setSecantType(tType);
}

void DOTk_SecantLeftPreconditionerFactory::setSecantType(dotk::types::invhessian_t aType)
{
    m_SecantType = aType;
}

}
