/*
 * DOTk_HessianFactory.hpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_HESSIANFACTORY_HPP_
#define DOTK_HESSIANFACTORY_HPP_

namespace dotk
{

class DOTk_SecondOrderOperator;
class DOTk_AssemblyManager;
class DOTk_OptimizationDataMng;

class DOTk_HessianFactory
{
public:
    DOTk_HessianFactory();
    explicit DOTk_HessianFactory(dotk::types::hessian_t type_);
    ~DOTk_HessianFactory();

    void setDefaultSecantSotrage(size_t val_);
    size_t getDefaultSecantSotrage() const;
    void setFactoryType(dotk::types::hessian_t type_);
    dotk::types::hessian_t getFactoryType() const;

    void buildFullSpaceHessian(std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildReducedSpaceHessian(std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildLbfgsHessian(size_t secant_storage_,
                           const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                           std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildLdfpHessian(size_t secant_storage_,
                          const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                          std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildLsr1Hessian(size_t secant_storage_,
                          const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                          std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildSr1Hessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                         std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildDfpHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                         std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);
    void buildBarzilaiBorweinHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                     std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_);

    void build(const dotk::DOTk_OptimizationDataMng * const mng_,
               std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & operator_,
               size_t secant_storage_ = 0);

private:
    size_t checkSecantStorageInput(size_t secant_storage_);

private:
    size_t mDefaultSecantSotrage;
    dotk::types::hessian_t mFactoryType;

private:
    DOTk_HessianFactory(const dotk::DOTk_HessianFactory &);
    dotk::DOTk_HessianFactory & operator=(const dotk::DOTk_HessianFactory &);
};

}

#endif /* DOTK_HESSIANFACTORY_HPP_ */
