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

template<typename ScalarType>
class Vector;

class DOTk_DFPHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    explicit DOTk_DFPHessian(const dotk::Vector<Real> & vector_);
    virtual ~DOTk_DFPHessian();

    const std::shared_ptr<dotk::Vector<Real> > & getDeltaGrad() const;
    const std::shared_ptr<dotk::Vector<Real> > & getDeltaPrimal() const;

    void getHessian(const std::shared_ptr<dotk::Vector<Real> > & vec_,
                    const std::shared_ptr<dotk::Vector<Real> > & hess_times_vec_);
    virtual void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::shared_ptr<dotk::Vector<Real> > & vec_,
                       const std::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;
    std::shared_ptr<dotk::Vector<Real> > m_HessTimesVec;

private:
    DOTk_DFPHessian(const dotk::DOTk_DFPHessian &);
    dotk::DOTk_DFPHessian operator=(const dotk::DOTk_DFPHessian &);
};

}

#endif /* DOTK_DFPHESSIAN_HPP_ */
