/*
 * DOTk_HessianFactory.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iostream>

#include "vector.hpp"
#include "DOTk_SR1Hessian.hpp"
#include "DOTk_DFPHessian.hpp"
#include "DOTk_LDFPHessian.hpp"
#include "DOTk_LSR1Hessian.hpp"
#include "DOTk_LBFGSHessian.hpp"
#include "DOTk_HessianFactory.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_UserDefinedHessian.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_BarzilaiBorweinHessian.hpp"
#include "DOTk_UserDefinedHessianTypeCNP.hpp"

namespace dotk
{

DOTk_HessianFactory::DOTk_HessianFactory() :
        mDefaultSecantSotrage(4),
        mFactoryType(dotk::types::HESSIAN_DISABLED)
{
}

DOTk_HessianFactory::DOTk_HessianFactory(dotk::types::hessian_t type_) :
        mDefaultSecantSotrage(4),
        mFactoryType(type_)
{
}

DOTk_HessianFactory::~DOTk_HessianFactory()
{
}

void DOTk_HessianFactory::setDefaultSecantSotrage(size_t val_)
{
    mDefaultSecantSotrage = val_;
}

size_t DOTk_HessianFactory::getDefaultSecantSotrage() const
{
    return (mDefaultSecantSotrage);
}

void DOTk_HessianFactory::setFactoryType(dotk::types::hessian_t type_)
{
    mFactoryType = type_;
}

dotk::types::hessian_t DOTk_HessianFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_HessianFactory::buildFullSpaceHessian(std::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    this->setFactoryType(dotk::types::USER_DEFINED_HESS_TYPE_CNP);
    operator_.reset(new dotk::DOTk_UserDefinedHessianTypeCNP);
}

void DOTk_HessianFactory::buildReducedSpaceHessian(std::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    this->setFactoryType(dotk::types::USER_DEFINED_HESS);
    operator_.reset(new dotk::DOTk_UserDefinedHessian);
}

void DOTk_HessianFactory::buildLbfgsHessian(size_t secant_storage_,
                                            const dotk::Vector<Real> & vector_,
                                            std::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    this->setFactoryType(dotk::types::LBFGS_HESS);
    size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
    operator_.reset(new dotk::DOTk_LBFGSHessian(vector_, secant_storage));
}

void DOTk_HessianFactory::buildLdfpHessian(size_t secant_storage_,
                                           const dotk::Vector<Real> & vector_,
                                           std::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    this->setFactoryType(dotk::types::LDFP_HESS);
    size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
    operator_.reset(new dotk::DOTk_LDFPHessian(vector_, secant_storage));
}

void DOTk_HessianFactory::buildLsr1Hessian(size_t secant_storage_,
                                           const dotk::Vector<Real> & vector_,
                                           std::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    this->setFactoryType(dotk::types::LSR1_HESS);
    size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
    operator_.reset(new dotk::DOTk_LSR1Hessian(vector_, secant_storage));
}

void DOTk_HessianFactory::buildSr1Hessian(const dotk::Vector<Real> & vector_,
                                          std::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    this->setFactoryType(dotk::types::SR1_HESS);
    operator_.reset(new dotk::DOTk_SR1Hessian(vector_));
}

void DOTk_HessianFactory::buildDfpHessian(const dotk::Vector<Real> & vector_,
                                          std::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    this->setFactoryType(dotk::types::DFP_HESS);
    operator_.reset(new dotk::DOTk_DFPHessian(vector_));
}

void DOTk_HessianFactory::buildBarzilaiBorweinHessian(const dotk::Vector<Real> & vector_,
                                                      std::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_)
{
    this->setFactoryType(dotk::types::BARZILAIBORWEIN_HESS);
    operator_.reset(new dotk::DOTk_BarzilaiBorweinHessian(vector_));
}

void DOTk_HessianFactory::build(const dotk::DOTk_OptimizationDataMng * const mng_,
                                std::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_,
                                size_t secant_storage_)
{
    switch(this->getFactoryType())
    {
        case dotk::types::LBFGS_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
            operator_.reset(new dotk::DOTk_LBFGSHessian(*mng_->getTrialStep(), secant_storage));
            break;
        }
        case dotk::types::LDFP_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
            operator_.reset(new dotk::DOTk_LDFPHessian(*mng_->getTrialStep(), secant_storage));
            break;
        }
        case dotk::types::LSR1_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(secant_storage_);
            operator_.reset(new dotk::DOTk_LSR1Hessian(*mng_->getTrialStep(), secant_storage));
            break;
        }
        case dotk::types::SR1_HESS:
        {
            operator_.reset(new dotk::DOTk_SR1Hessian(*mng_->getTrialStep()));
            break;
        }
        case dotk::types::DFP_HESS:
        {
            operator_.reset(new dotk::DOTk_DFPHessian(*mng_->getTrialStep()));
            break;
        }
        case dotk::types::BARZILAIBORWEIN_HESS:
        {
            operator_.reset(new dotk::DOTk_BarzilaiBorweinHessian(*mng_->getTrialStep()));
            break;
        }
        case dotk::types::USER_DEFINED_HESS:
        {
            operator_.reset(new dotk::DOTk_UserDefinedHessian);
            break;
        }
        case dotk::types::HESSIAN_DISABLED:
        {
            break;
        }
        case dotk::types::USER_DEFINED_HESS_TYPE_CNP:
        default:
        {
            std::cout << "\nDOTk WARNING: Invalid hessian operator type, Default Hessian operator set to LBFGS "
                    << "and the secant storage will be set to 4.\n" << std::flush;
            operator_.reset(new dotk::DOTk_LBFGSHessian(*mng_->getTrialStep(), this->getDefaultSecantSotrage()));
            break;
        }
    }
}

size_t DOTk_HessianFactory::checkSecantStorageInput(size_t secant_storage_)
{
    size_t secant_storage = secant_storage_;
    if(secant_storage_ <= 0)
    {
        std::cout << "\nDOTk WARNING: Invalid secant storage input. Default secant storage will be set to "
                << this->getDefaultSecantSotrage() << ".\n\n" << std::flush;
        secant_storage = this->getDefaultSecantSotrage();
    }
    return (secant_storage);
}

}
