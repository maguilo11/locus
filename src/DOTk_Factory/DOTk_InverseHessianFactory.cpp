/*
 * DOTk_InverseHessianFactory.cpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iostream>

#include "vector.hpp"
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

DOTk_InverseHessianFactory::DOTk_InverseHessianFactory(dotk::types::invhessian_t aType) :
        mDefaultSecantSotrage(4),
        mFactoryType(aType)
{
}

DOTk_InverseHessianFactory::~DOTk_InverseHessianFactory()
{
}

void DOTk_InverseHessianFactory::setDefaultSecantSotrage(size_t aInput)
{
    mDefaultSecantSotrage = aInput;
}

size_t DOTk_InverseHessianFactory::getDefaultSecantSotrage() const
{
    return (mDefaultSecantSotrage);
}

void DOTk_InverseHessianFactory::setFactoryType(dotk::types::invhessian_t aType)
{
    mFactoryType = aType;
}

dotk::types::invhessian_t DOTk_InverseHessianFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_InverseHessianFactory::buildLbfgsInvHessian
(size_t aSecantStorageSize,
 const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
    aOutput = std::make_shared<dotk::DOTk_LBFGSInvHessian>(aVector, secant_storage);
    this->setFactoryType(dotk::types::LBFGS_INV_HESS);
}

void DOTk_InverseHessianFactory::buildLdfpInvHessian
(size_t aSecantStorageSize,
 const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
    aOutput = std::make_shared<dotk::DOTk_LDFPInvHessian>(aVector, secant_storage);
    this->setFactoryType(dotk::types::LDFP_INV_HESS);
}

void DOTk_InverseHessianFactory::buildLsr1InvHessian
(size_t aSecantStorageSize,
 const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
    aOutput = std::make_shared<dotk::DOTk_LSR1InvHessian>(aVector, secant_storage);
    this->setFactoryType(dotk::types::LSR1_INV_HESS);
}

void DOTk_InverseHessianFactory::buildSr1InvHessian
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_SR1InvHessian>(aVector);
    this->setFactoryType(dotk::types::SR1_INV_HESS);
}

void DOTk_InverseHessianFactory::buildBfgsInvHessian
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_BFGSInvHessian>(aVector);
    this->setFactoryType(dotk::types::BFGS_INV_HESS);
}

void DOTk_InverseHessianFactory::buildBarzilaiBorweinInvHessian
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_BarzilaiBorweinInvHessian>(aVector);
    this->setFactoryType(dotk::types::BARZILAIBORWEIN_INV_HESS);
}

void DOTk_InverseHessianFactory::build
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput,
 size_t aSecantStorageSize)
{
    switch(this->getFactoryType())
    {
        case dotk::types::LBFGS_INV_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
            aOutput = std::make_shared<dotk::DOTk_LBFGSInvHessian>(aVector, secant_storage);
            break;
        }
        case dotk::types::LDFP_INV_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
            aOutput = std::make_shared<dotk::DOTk_LDFPInvHessian>(aVector, secant_storage);
            break;
        }
        case dotk::types::LSR1_INV_HESS:
        {
            size_t secant_storage = this->checkSecantStorageInput(aSecantStorageSize);
            aOutput = std::make_shared<dotk::DOTk_LSR1InvHessian>(aVector, secant_storage);
            break;
        }
        case dotk::types::SR1_INV_HESS:
        {
            aOutput = std::make_shared<dotk::DOTk_SR1InvHessian>(aVector);
            break;
        }
        case dotk::types::BFGS_INV_HESS:
        {
            aOutput = std::make_shared<dotk::DOTk_BFGSInvHessian>(aVector);
            break;
        }
        case dotk::types::BARZILAIBORWEIN_INV_HESS:
        {
            aOutput = std::make_shared<dotk::DOTk_BarzilaiBorweinInvHessian>(aVector);
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
            size_t tSecantSotrageSize = this->getDefaultSecantSotrage();
            aOutput = std::make_shared<dotk::DOTk_LBFGSInvHessian>(aVector, tSecantSotrageSize);
            break;
        }
    }
}

size_t DOTk_InverseHessianFactory::checkSecantStorageInput(size_t aSecantStorageSize)
{
    size_t secant_storage = aSecantStorageSize;
    if(aSecantStorageSize <= 0)
    {
        std::cout << "\nDOTk WARNING: Invalid secant storage input. Default secant storage will be set to "
                  << this->getDefaultSecantSotrage() << ".\n\n" << std::flush;
        secant_storage = this->getDefaultSecantSotrage();
    }
    return(secant_storage);
}

}
