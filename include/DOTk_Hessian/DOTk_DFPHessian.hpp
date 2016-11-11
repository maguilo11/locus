/*
 * DOTk_DFPHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DFPHESSIAN_HPP_
#define DOTK_DFPHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_DFPHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    explicit DOTk_DFPHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_DFPHessian();

    const std::tr1::shared_ptr<dotk::vector<Real> > & getDeltaGrad() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getDeltaPrimal() const;

    void getHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                    const std::tr1::shared_ptr<dotk::vector<Real> > & hess_times_vec_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vec_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaPrimal;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaGradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_HessTimesVec;

private:
    DOTk_DFPHessian(const dotk::DOTk_DFPHessian &);
    dotk::DOTk_DFPHessian operator=(const dotk::DOTk_DFPHessian &);
};

}

#endif /* DOTK_DFPHESSIAN_HPP_ */