/*
 * DOTk_InverseHessianFactory.hpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INVERSEHESSIANFACTORY_HPP_
#define DOTK_INVERSEHESSIANFACTORY_HPP_

#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_SecondOrderOperator;

class DOTk_InverseHessianFactory
{
public:
    DOTk_InverseHessianFactory();
    explicit DOTk_InverseHessianFactory(dotk::types::invhessian_t type_);
    ~DOTk_InverseHessianFactory();

    void setDefaultSecantSotrage(size_t val_);
    size_t getDefaultSecantSotrage() const;
    void setFactoryType(dotk::types::invhessian_t type_);
    dotk::types::invhessian_t getFactoryType() const;

    void buildLbfgsInvHessian(size_t secant_storage_,
                              const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                              std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildLdfpInvHessian(size_t secant_storage_,
                             const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                             std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildLsr1InvHessian(size_t secant_storage_,
                             const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                             std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildSr1InvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                            std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildBfgsInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                             std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildBarzilaiBorweinInvHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                        std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);

    void build(const std::tr1::shared_ptr<dotk::vector<Real> > & vec_template_,
               std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_,
               size_t secant_storage_ = 0);

private:
    size_t mDefaultSecantSotrage;
    dotk::types::invhessian_t mFactoryType;

private:
    size_t checkSecantStorageInput(size_t secant_storage_);

private:
    DOTk_InverseHessianFactory(const dotk::DOTk_InverseHessianFactory &);
    dotk::DOTk_InverseHessianFactory & operator=(const dotk::DOTk_InverseHessianFactory &);
};

}

#endif /* DOTK_INVERSEHESSIANFACTORY_HPP_ */
