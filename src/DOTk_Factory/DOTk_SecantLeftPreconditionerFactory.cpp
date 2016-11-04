/*
 * DOTk_SecantLeftPreconditionerFactory.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

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
(size_t secant_storage_,
 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_)
{
    left_prec_.reset(new dotk::DOTK_SecantLeftPreconditioner(vector_, dotk::types::LDFP_INV_HESS, secant_storage_));
    this->setSecantType(dotk::types::LDFP_INV_HESS);
}

void DOTk_SecantLeftPreconditionerFactory::buildLsr1SecantPreconditioner
(size_t secant_storage_,
 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_)
{
    left_prec_.reset(new dotk::DOTK_SecantLeftPreconditioner(vector_, dotk::types::LSR1_INV_HESS, secant_storage_));
    this->setSecantType(dotk::types::LSR1_INV_HESS);
}

void DOTk_SecantLeftPreconditionerFactory::buildLbfgsSecantPreconditioner
(size_t secant_storage_,
 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_)
{
    left_prec_.reset(new dotk::DOTK_SecantLeftPreconditioner(vector_, dotk::types::LBFGS_INV_HESS, secant_storage_));
    this->setSecantType(dotk::types::LBFGS_INV_HESS);
}

void DOTk_SecantLeftPreconditionerFactory::buildBfgsSecantPreconditioner
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_)
{
    left_prec_.reset(new dotk::DOTK_SecantLeftPreconditioner(vector_, dotk::types::BFGS_INV_HESS));
    this->setSecantType(dotk::types::BFGS_INV_HESS);
}

void DOTk_SecantLeftPreconditionerFactory::buildSr1SecantPreconditioner
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_)
{
    left_prec_.reset(new dotk::DOTK_SecantLeftPreconditioner(vector_, dotk::types::SR1_INV_HESS));
    this->setSecantType(dotk::types::SR1_INV_HESS);
}

void DOTk_SecantLeftPreconditionerFactory::buildBarzilaiBorweinSecantPreconditioner
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & left_prec_)
{
    left_prec_.reset(new dotk::DOTK_SecantLeftPreconditioner(vector_, dotk::types::BARZILAIBORWEIN_INV_HESS));
    this->setSecantType(dotk::types::BARZILAIBORWEIN_INV_HESS);
}

void DOTk_SecantLeftPreconditionerFactory::setSecantType(dotk::types::invhessian_t type_)
{
    m_SecantType = type_;
}

}
