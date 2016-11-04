/*
 * DOTk_Hessian.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_HESSIAN_HPP_
#define DOTK_HESSIAN_HPP_

#include "DOTk_LinearOperator.hpp"

namespace dotk
{

class DOTk_SecondOrderOperator;
class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_Hessian : public dotk::DOTk_LinearOperator
{
public:
    DOTk_Hessian(); // DEFAULT SET TO REDUCED HESSIAN (I.E. USER DEFINED HESSIAN FOR A REDUCED SPACE FORMULATION)
    explicit DOTk_Hessian(const std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> & hessian_);
    virtual ~DOTk_Hessian();

    dotk::types::hessian_t hessianType() const;

    void setFullSpaceHessian();
    void setReducedSpaceHessian();
    void setSr1Hessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    void setDfpHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    void setBarzilaiBorweinHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    void setLbfgsHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_, size_t secant_storage_ = 2);
    void setLdfpHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_, size_t secant_storage_ = 2);
    void setLsr1Hessian(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_, size_t secant_storage_ = 2);

    virtual void setNumOtimizationItrDone(size_t itr_);
    virtual void updateLimitedMemoryStorage(bool update_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & output_);

private:
    dotk::types::hessian_t m_Type;
    std::tr1::shared_ptr<dotk::DOTk_SecondOrderOperator> m_Hessian;

private:
    DOTk_Hessian(const dotk::DOTk_Hessian &);
    dotk::DOTk_Hessian & operator=(const dotk::DOTk_Hessian & rhs_);
};

}

#endif /* DOTK_HESSIAN_HPP_ */
