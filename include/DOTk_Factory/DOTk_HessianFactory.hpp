/*
 * DOTk_HessianFactory.hpp
 *
 *  Created on: Oct 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_HESSIANFACTORY_HPP_
#define DOTK_HESSIANFACTORY_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_SecondOrderOperator;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_HessianFactory
{
public:
    DOTk_HessianFactory();
    explicit DOTk_HessianFactory(dotk::types::hessian_t aType);
    ~DOTk_HessianFactory();

    void setDefaultSecantSotrage(size_t aInput);
    size_t getDefaultSecantSotrage() const;
    void setFactoryType(dotk::types::hessian_t aType);
    dotk::types::hessian_t getFactoryType() const;

    void buildFullSpaceHessian(std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildReducedSpaceHessian(std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildLbfgsHessian(size_t aSecantStorageSize,
                           const dotk::Vector<Real> & vector_,
                           std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildLdfpHessian(size_t aSecantStorageSize,
                          const dotk::Vector<Real> & vector_,
                          std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildLsr1Hessian(size_t aSecantStorageSize,
                          const dotk::Vector<Real> & vector_,
                          std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildSr1Hessian(const dotk::Vector<Real> & vector_,
                         std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildDfpHessian(const dotk::Vector<Real> & vector_,
                         std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildBarzilaiBorweinHessian(const dotk::Vector<Real> & vector_,
                                     std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);

    void build(const dotk::DOTk_OptimizationDataMng * const aMng,
               std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput,
               size_t aSecantStorageSize = 0);

private:
    size_t checkSecantStorageInput(size_t aSecantStorageSize);

private:
    size_t mDefaultSecantSotrage;
    dotk::types::hessian_t mFactoryType;

private:
    DOTk_HessianFactory(const dotk::DOTk_HessianFactory &);
    dotk::DOTk_HessianFactory & operator=(const dotk::DOTk_HessianFactory &);
};

}

#endif /* DOTK_HESSIANFACTORY_HPP_ */
