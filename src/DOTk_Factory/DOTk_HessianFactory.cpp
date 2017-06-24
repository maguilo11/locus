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

DOTk_HessianFactory::DOTk_HessianFactory(dotk::types::hessian_t aType) :
        mDefaultSecantSotrage(4),
        mFactoryType(aType)
{
}

DOTk_HessianFactory::~DOTk_HessianFactory()
{
}

void DOTk_HessianFactory::setDefaultSecantSotrage(size_t aInput)
{
    mDefaultSecantSotrage = aInput;
}

size_t DOTk_HessianFactory::getDefaultSecantSotrage() const
{
    return (mDefaultSecantSotrage);
}

void DOTk_HessianFactory::setFactoryType(dotk::types::hessian_t aType)
{
    mFactoryType = aType;
}

dotk::types::hessian_t DOTk_HessianFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_HessianFactory::buildFullSpaceHessian(std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    this->setFactoryType(dotk::types::USER_DEFINED_HESS_TYPE_CNP);
    aOutput = std::make_shared<dotk::DOTk_UserDefinedHessianTypeCNP>();
}

void DOTk_HessianFactory::buildReducedSpaceHessian(std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    this->setFactoryType(dotk::types::USER_DEFINED_HESS);
    aOutput = std::make_shared<dotk::DOTk_UserDefinedHessian>();
}

void DOTk_HessianFactory::buildLbfgsHessian(size_t aSecantStorageSize,
                                            const dotk::Vector<Real> & vector_,
                                            std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    this->setFactoryType(dotk::types::LBFGS_HESS);
    size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
    aOutput = std::make_shared<dotk::DOTk_LBFGSHessian>(vector_, secant_storage);
}

void DOTk_HessianFactory::buildLdfpHessian(size_t aSecantStorageSize,
                                           const dotk::Vector<Real> & vector_,
                                           std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    this->setFactoryType(dotk::types::LDFP_HESS);
    size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
    aOutput = std::make_shared<dotk::DOTk_LDFPHessian>(vector_, secant_storage);
}

void DOTk_HessianFactory::buildLsr1Hessian(size_t aSecantStorageSize,
                                           const dotk::Vector<Real> & vector_,
                                           std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    this->setFactoryType(dotk::types::LSR1_HESS);
    size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
    aOutput = std::make_shared<dotk::DOTk_LSR1Hessian>(vector_, secant_storage);
}

void DOTk_HessianFactory::buildSr1Hessian(const dotk::Vector<Real> & vector_,
                                          std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    this->setFactoryType(dotk::types::SR1_HESS);
    aOutput = std::make_shared<dotk::DOTk_SR1Hessian>(vector_);
}

void DOTk_HessianFactory::buildDfpHessian(const dotk::Vector<Real> & vector_,
                                          std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    this->setFactoryType(dotk::types::DFP_HESS);
    aOutput = std::make_shared<dotk::DOTk_DFPHessian>(vector_);
}

void DOTk_HessianFactory::buildBarzilaiBorweinHessian(const dotk::Vector<Real> & vector_,
                                                      std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    this->setFactoryType(dotk::types::BARZILAIBORWEIN_HESS);
    aOutput = std::make_shared<dotk::DOTk_BarzilaiBorweinHessian>(vector_);
}

void DOTk_HessianFactory::build(const dotk::DOTk_OptimizationDataMng * const aMng,
                                std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput,
                                size_t aSecantStorageSize)
{
    switch(this->getFactoryType())
    {
        case dotk::types::LBFGS_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
            aOutput = std::make_shared<dotk::DOTk_LBFGSHessian>(*aMng->getTrialStep(), secant_storage);
            break;
        }
        case dotk::types::LDFP_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
            aOutput = std::make_shared<dotk::DOTk_LDFPHessian>(*aMng->getTrialStep(), secant_storage);
            break;
        }
        case dotk::types::LSR1_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
            aOutput = std::make_shared<dotk::DOTk_LSR1Hessian>(*aMng->getTrialStep(), secant_storage);
            break;
        }
        case dotk::types::SR1_HESS:
        {
            aOutput = std::make_shared<dotk::DOTk_SR1Hessian>(*aMng->getTrialStep());
            break;
        }
        case dotk::types::DFP_HESS:
        {
            aOutput = std::make_shared<dotk::DOTk_DFPHessian>(*aMng->getTrialStep());
            break;
        }
        case dotk::types::BARZILAIBORWEIN_HESS:
        {
            aOutput = std::make_shared<dotk::DOTk_BarzilaiBorweinHessian>(*aMng->getTrialStep());
            break;
        }
        case dotk::types::USER_DEFINED_HESS:
        {
            aOutput = std::make_shared<dotk::DOTk_UserDefinedHessian>();
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
            size_t secant_storage = this->getDefaultSecantSotrage();
            aOutput = std::make_shared<dotk::DOTk_LBFGSHessian>(*aMng->getTrialStep(), secant_storage);
            break;
        }
    }
}

size_t DOTk_HessianFactory::checkSecantStorageInput(size_t aSecantStorageSize)
{
    size_t secant_storage = aSecantStorageSize;
    if(aSecantStorageSize <= 0)
    {
        std::cout << "\nDOTk WARNING: Invalid secant storage input. Default secant storage will be set to "
                << this->getDefaultSecantSotrage() << ".\n\n" << std::flush;
        secant_storage = this->getDefaultSecantSotrage();
    }
    return (secant_storage);
}

}
