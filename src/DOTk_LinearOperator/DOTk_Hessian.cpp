/*
 * DOTk_Hessian.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>

#include "vector.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_HessianFactory.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_Hessian::DOTk_Hessian() :
        dotk::DOTk_LinearOperator(dotk::types::HESSIAN_MATRIX),
        m_Type(dotk::types::HESSIAN_DISABLED),
        m_Hessian()
{
    this->setReducedSpaceHessian();
}

DOTk_Hessian::DOTk_Hessian(const std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & hessian_) :
        dotk::DOTk_LinearOperator(dotk::types::HESSIAN_MATRIX),
        m_Type(dotk::types::HESSIAN_DISABLED),
        m_Hessian(hessian_)
{
}

DOTk_Hessian::~DOTk_Hessian()
{
}

dotk::types::hessian_t DOTk_Hessian::hessianType() const
{
    return (m_Type);
}

void DOTk_Hessian::setNumOtimizationItrDone(size_t itr_)
{
    m_Hessian->setNumOptimizationItrDone(itr_);
}

void DOTk_Hessian::updateLimitedMemoryStorage(bool update_)
{
    m_Hessian->setUpdateSecondOrderOperator(true);
}

void DOTk_Hessian::setSr1Hessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_)
{
    dotk::DOTk_HessianFactory factory;
    factory.buildSr1Hessian(vector_, m_Hessian);
    m_Type = factory.getFactoryType();
}

void DOTk_Hessian::setDfpHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_)
{
    dotk::DOTk_HessianFactory factory;
    factory.buildDfpHessian(vector_, m_Hessian);
    m_Type = factory.getFactoryType();
}

void DOTk_Hessian::setReducedSpaceHessian()
{
    dotk::DOTk_HessianFactory factory;
    factory.buildReducedSpaceHessian(m_Hessian);
    m_Type = factory.getFactoryType();
}

void DOTk_Hessian::setFullSpaceHessian()
{
    dotk::DOTk_HessianFactory factory;
    factory.buildFullSpaceHessian(m_Hessian);
    m_Type = factory.getFactoryType();
}

void DOTk_Hessian::setBarzilaiBorweinHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_)
{
    dotk::DOTk_HessianFactory factory;
    factory.buildBarzilaiBorweinHessian(vector_, m_Hessian);
    m_Type = factory.getFactoryType();
}

void DOTk_Hessian::setLbfgsHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_, size_t secant_storage_)
{
    dotk::DOTk_HessianFactory factory;
    factory.buildLbfgsHessian(secant_storage_, vector_, m_Hessian);
    m_Type = factory.getFactoryType();
}

void DOTk_Hessian::setLdfpHessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_, size_t secant_storage_)
{
    dotk::DOTk_HessianFactory factory;
    factory.buildLdfpHessian(secant_storage_, vector_, m_Hessian);
    m_Type = factory.getFactoryType();
}

void DOTk_Hessian::setLsr1Hessian(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_, size_t secant_storage_)
{
    dotk::DOTk_HessianFactory factory;
    factory.buildLsr1Hessian(secant_storage_, vector_, m_Hessian);
    m_Type = factory.getFactoryType();
}

void DOTk_Hessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                         const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                         const std::tr1::shared_ptr<dotk::Vector<Real> > & output_)
{
    assert(m_Hessian.use_count() > 0);
    m_Hessian->apply(mng_, vector_, output_);
}

}
