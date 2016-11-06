/*
 * DOTk_InverseHessianFactory.cpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iostream>

#include "DOTk_SR1InvHessian.hpp"
#include "DOTk_LDFPInvHessian.hpp"
#include "DOTk_LSR1InvHessian.hpp"
#include "DOTk_BFGSInvHessian.hpp"
#include "DOTk_LBFGSInvHessian.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_InverseHessianFactory.hpp"
#include "DOTk_BarzilaiBorweinInvHessian.hpp"

namespace dotk
{

DOTk_InverseHessianFactory::DOTk_InverseHessianFactory() :
        mDefaultSecantSotrage(4),
        mFactoryType(dotk::types::INV_HESS_DISABLED)
{
}

DOTk_InverseHessianFactory::DOTk_InverseHessianFactory(dotk::types::invhessian_t type_) :
        mDefaultSecantSotrage(4),
        mFactoryType(type_)
{
}

DOTk_InverseHessianFactory::~DOTk_InverseHessianFactory()
{
}

void DOTk_InverseHessianFactory::setDefaultSecantSotrage(size_t val_)
{
    mDefaultSecantSotrage = val_;
}

size_t DOTk_InverseHessianFactory::getDefaultSecantSotrage() const
{
    return (mDefaultSecantSotrage);
}

void DOTk_InverseHessianFactory::setFactoryType(dotk::types::invhessian_t type_)
{
    mFactoryType = type_;
}

dotk::types::invhessian_t DOTk_InverseHessianFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_InverseHessianFactory::buildLbfgsInvHessian
(size_t secant_storage_,
 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
    operator_.reset(new dotk::DOTk_LBFGSInvHessian(vector_, secant_storage));
    this->setFactoryType(dotk::types::LBFGS_INV_HESS);
}

void DOTk_InverseHessianFactory::buildLdfpInvHessian
(size_t secant_storage_,
 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
    operator_.reset(new dotk::DOTk_LDFPInvHessian(vector_, secant_storage));
    this->setFactoryType(dotk::types::LDFP_INV_HESS);
}

void DOTk_InverseHessianFactory::buildLsr1InvHessian
(size_t secant_storage_,
 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
    operator_.reset(new dotk::DOTk_LSR1InvHessian(vector_, secant_storage));
    this->setFactoryType(dotk::types::LSR1_INV_HESS);
}

void DOTk_InverseHessianFactory::buildSr1InvHessian
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    operator_.reset(new dotk::DOTk_SR1InvHessian(vector_));
    this->setFactoryType(dotk::types::SR1_INV_HESS);
}

void DOTk_InverseHessianFactory::buildBfgsInvHessian
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    operator_.reset(new dotk::DOTk_BFGSInvHessian(vector_));
    this->setFactoryType(dotk::types::BFGS_INV_HESS);
}

void DOTk_InverseHessianFactory::buildBarzilaiBorweinInvHessian
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    operator_.reset(new dotk::DOTk_BarzilaiBorweinInvHessian(vector_));
    this->setFactoryType(dotk::types::BARZILAIBORWEIN_INV_HESS);
}

void DOTk_InverseHessianFactory::build
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_,
 size_t secant_storage_)
{
    switch(this->getFactoryType())
    {
        case dotk::types::LBFGS_INV_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
            operator_.reset(new dotk::DOTk_LBFGSInvHessian(vector_, secant_storage));
            break;
        }
        case dotk::types::LDFP_INV_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
            operator_.reset(new dotk::DOTk_LDFPInvHessian(vector_, secant_storage));
            break;
        }
        case dotk::types::LSR1_INV_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
            operator_.reset(new dotk::DOTk_LSR1InvHessian(vector_, secant_storage));
            break;
        }
        case dotk::types::SR1_INV_HESS:
        {
            operator_.reset(new dotk::DOTk_SR1InvHessian(vector_));
            break;
        }
        case dotk::types::BFGS_INV_HESS:
        {
            operator_.reset(new dotk::DOTk_BFGSInvHessian(vector_));
            break;
        }
        case dotk::types::BARZILAIBORWEIN_INV_HESS:
        {
            operator_.reset(new dotk::DOTk_BarzilaiBorweinInvHessian(vector_));
            break;
        }
        case dotk::types::INV_HESS_DISABLED:
        {
            break;
        }
        default:
        {
            std::cout << "\nDOTk WARNING: Invalid inverse hessian operator type, Default inverse Hessian operator set to LBFGS "
                    << "and the secant storage will be set to 4.\n" << std::flush;
            operator_.reset(new dotk::DOTk_LBFGSInvHessian(vector_, this->getDefaultSecantSotrage()));
            break;
        }
    }
}

size_t DOTk_InverseHessianFactory::checkSecantStorageInput(size_t secant_storage_)
{
    size_t secant_storage = secant_storage_;
    if(secant_storage_ <= 0)
    {
        std::cout << "\nDOTk WARNING: Invalid secant storage input. Default secant storage will be set to "
                  << this->getDefaultSecantSotrage() << ".\n\n" << std::flush;
        secant_storage = this->getDefaultSecantSotrage();
    }
    return(secant_storage);
}

}
