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

template<class Type>
class vector;

class DOTk_SR1Hessian : public dotk::DOTk_SecondOrderOperator
{
public:
    explicit DOTk_SR1Hessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_SR1Hessian();

    const std::tr1::shared_ptr<dotk::vector<Real> > & getDeltaGrad() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getDeltaPrimal() const;

    void getHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                    const std::tr1::shared_ptr<dotk::vector<Real> > & hess_times_vector_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vector_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaPrimal;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaGradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_HessTimesVec;


private:
    DOTk_SR1Hessian(const dotk::DOTk_SR1Hessian &);
    dotk::DOTk_SR1Hessian operator=(const dotk::DOTk_SR1Hessian &);
};

}

#endif /* DOTK_SR1HESSIAN_HPP_ */
