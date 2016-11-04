/*
 * DOTk_NumericallyDifferentiatedHessian.hpp
 *
 *  Created on: Oct 8, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NUMERICALLYDIFFERENTIATEDHESSIAN_HPP_
#define DOTK_NUMERICALLYDIFFERENTIATEDHESSIAN_HPP_

#include "DOTk_LinearOperator.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_Functor;
class DOTk_OptimizationDataMng;
class DOTk_NumericalDifferentiation;

template<class Type>
class vector;
template<class Type>
class DOTk_ObjectiveFunction;
template<class Type>
class DOTk_EqualityConstraint;

class NumericallyDifferentiatedHessian : public dotk::DOTk_LinearOperator
{
public:
    NumericallyDifferentiatedHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_);
    NumericallyDifferentiatedHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                     const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~NumericallyDifferentiatedHessian();

    void setForwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_ = 1e-6);
    void setBackwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_ = 1e-6);
    void setCentralDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_ = 1e-6);
    void setSecondOrderForwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_ = 1e-6);
    void setThirdOrderForwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_ = 1e-6);
    void setThirdOrderBackwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, Real epsilon_ = 1e-6);

    void apply(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
               const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_,
               const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
               const std::tr1::shared_ptr<dotk::vector<Real> > & output_);
    void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
               const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
               const std::tr1::shared_ptr<dotk::vector<Real> > & output_);

    void setNumOtimizationItrDone(size_t itr_);
    void updateLimitedMemoryStorage(bool update_);

private:
    std::tr1::shared_ptr<dotk::DOTk_Functor> m_GradientFunctor;
    std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> m_NumericalDifferentiation;
};

}

#endif /* DOTK_NUMERICALLYDIFFERENTIATEDHESSIAN_HPP_ */
