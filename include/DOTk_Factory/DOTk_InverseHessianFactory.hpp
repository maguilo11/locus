/*
 * DOTk_InverseHessianFactory.hpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INVERSEHESSIANFACTORY_HPP_
#define DOTK_INVERSEHESSIANFACTORY_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_SecondOrderOperator;

template<typename ScalarType>
class Vector;

class DOTk_InverseHessianFactory
{
public:
    DOTk_InverseHessianFactory();
    explicit DOTk_InverseHessianFactory(dotk::types::invhessian_t aType);
    ~DOTk_InverseHessianFactory();

    void setDefaultSecantSotrage(size_t aInput);
    size_t getDefaultSecantSotrage() const;
    void setFactoryType(dotk::types::invhessian_t aType);
    dotk::types::invhessian_t getFactoryType() const;

    void buildLbfgsInvHessian(size_t aSecantStorageSize,
                              const std::shared_ptr<dotk::Vector<Real> > & aVector,
                              std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildLdfpInvHessian(size_t aSecantStorageSize,
                             const std::shared_ptr<dotk::Vector<Real> > & aVector,
                             std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildLsr1InvHessian(size_t aSecantStorageSize,
                             const std::shared_ptr<dotk::Vector<Real> > & aVector,
                             std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildSr1InvHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                            std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildBfgsInvHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                             std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);
    void buildBarzilaiBorweinInvHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                        std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput);

    void build(const std::shared_ptr<dotk::Vector<Real> > & aVector,
               std::shared_ptr<dotk::DOTk_SecondOrderOperator> & aOutput,
               size_t aSecantStorageSize = 0);

private:
    size_t mDefaultSecantSotrage;
    dotk::types::invhessian_t mFactoryType;

private:
    size_t checkSecantStorageInput(size_t aSecantStorageSize);

private:
    DOTk_InverseHessianFactory(const dotk::DOTk_InverseHessianFactory &);
    dotk::DOTk_InverseHessianFactory & operator=(const dotk::DOTk_InverseHessianFactory &);
};

}

#endif /* DOTK_INVERSEHESSIANFACTORY_HPP_ */
