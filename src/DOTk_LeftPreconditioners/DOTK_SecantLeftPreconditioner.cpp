/*
 * DOTK_SecantLeftPreconditioner.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_InverseHessianFactory.hpp"
#include "DOTK_SecantLeftPreconditioner.hpp"

namespace dotk
{

DOTK_SecantLeftPreconditioner::DOTK_SecantLeftPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                             dotk::types::invhessian_t type_,
                                                             size_t secant_storage_) :
        dotk::DOTk_LeftPreconditioner(dotk::types::SECANT_LEFT_PRECONDITIONER),
        m_SecantLeftPrecType(dotk::types::INV_HESS_DISABLED),
        m_SecantLeftPrec()
{
    dotk::DOTk_InverseHessianFactory factory(type_);
    factory.build(vector_, m_SecantLeftPrec, secant_storage_);
}

DOTK_SecantLeftPreconditioner::~DOTK_SecantLeftPreconditioner()
{
}

dotk::types::invhessian_t DOTK_SecantLeftPreconditioner::getSecantLeftPrecType() const
{
    return (m_SecantLeftPrecType);
}

void DOTK_SecantLeftPreconditioner::setLbfgsPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                           size_t secant_storage_)
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildLbfgsInvHessian(secant_storage_, vector_, m_SecantLeftPrec);
    this->setSecantLeftPrecType(factory.getFactoryType());
}

void DOTK_SecantLeftPreconditioner::setLdfpPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                          size_t secant_storage_)
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildLdfpInvHessian(secant_storage_, vector_, m_SecantLeftPrec);
    this->setSecantLeftPrecType(factory.getFactoryType());
}

void DOTK_SecantLeftPreconditioner::setLsr1Preconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                          size_t secant_storage_)
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildLsr1InvHessian(secant_storage_, vector_, m_SecantLeftPrec);
    this->setSecantLeftPrecType(factory.getFactoryType());
}

void DOTK_SecantLeftPreconditioner::setBfgsPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_)
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildBfgsInvHessian(vector_, m_SecantLeftPrec);
    this->setSecantLeftPrecType(factory.getFactoryType());
}

void DOTK_SecantLeftPreconditioner::setSr1Preconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_)
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildSr1InvHessian(vector_, m_SecantLeftPrec);
    this->setSecantLeftPrecType(factory.getFactoryType());
}

void DOTK_SecantLeftPreconditioner::setBarzilaiBorweinPreconditioner(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_)
{
    dotk::DOTk_InverseHessianFactory factory;
    factory.buildBarzilaiBorweinInvHessian(vector_, m_SecantLeftPrec);
    this->setSecantLeftPrecType(factory.getFactoryType());
}

void DOTK_SecantLeftPreconditioner::setNumOptimizationItrDone(size_t itr_)
{
    dotk::DOTk_LeftPreconditioner::setNumOptimizationItrDone(itr_);
    m_SecantLeftPrec->setNumOptimizationItrDone(itr_);
}

void DOTK_SecantLeftPreconditioner::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_,
                                          const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                                          const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vec_)
{
    m_SecantLeftPrec->apply(opt_mng_, vec_, matrix_times_vec_);
}

void DOTK_SecantLeftPreconditioner::setSecantLeftPrecType(dotk::types::invhessian_t type_)
{
    m_SecantLeftPrecType = type_;
}

}
