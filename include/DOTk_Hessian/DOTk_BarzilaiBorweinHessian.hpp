/*
 * DOTk_BarzilaiBorweinHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_BARZILAIBORWEINHESSIAN_HPP_
#define DOTK_BARZILAIBORWEINHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_BarzilaiBorweinHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    explicit DOTk_BarzilaiBorweinHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_BarzilaiBorweinHessian();

    void computeDeltaPrimal(const std::tr1::shared_ptr<dotk::vector<Real> > & new_primal_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & old_primal_);
    void computeDeltaGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & new_gradient_,
                              const std::tr1::shared_ptr<dotk::vector<Real> > & old_gradient_);
    void getHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                    const std::tr1::shared_ptr<dotk::vector<Real> > & mat_times_vec_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & vec_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vec_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaPrimal;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DeltaGradient;


private:
    DOTk_BarzilaiBorweinHessian(const dotk::DOTk_BarzilaiBorweinHessian &);
    dotk::DOTk_BarzilaiBorweinHessian operator=(const dotk::DOTk_BarzilaiBorweinHessian &);
};


}

#endif /* DOTK_BARZILAIBORWEINHESSIAN_HPP_ */
