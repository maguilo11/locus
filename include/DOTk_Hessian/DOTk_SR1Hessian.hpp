/*
 * DOTk_SR1Hessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SR1HESSIAN_HPP_
#define DOTK_SR1HESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_SR1Hessian : public dotk::DOTk_SecondOrderOperator
{
public:
    explicit DOTk_SR1Hessian(const dotk::Vector<Real> & vector_);
    virtual ~DOTk_SR1Hessian();

    const std::shared_ptr<dotk::Vector<Real> > & getDeltaGrad() const;
    const std::shared_ptr<dotk::Vector<Real> > & getDeltaPrimal() const;

    void getHessian(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                    const std::shared_ptr<dotk::Vector<Real> > & hess_times_vector_);
    virtual void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vector_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;
    std::shared_ptr<dotk::Vector<Real> > m_HessTimesVec;


private:
    DOTk_SR1Hessian(const dotk::DOTk_SR1Hessian &);
    dotk::DOTk_SR1Hessian operator=(const dotk::DOTk_SR1Hessian &);
};

}

#endif /* DOTK_SR1HESSIAN_HPP_ */
