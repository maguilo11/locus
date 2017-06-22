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

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class DOTk_ObjectiveFunction;
template<typename ScalarType>
class DOTk_EqualityConstraint;

class NumericallyDifferentiatedHessian : public dotk::DOTk_LinearOperator
{
public:
    NumericallyDifferentiatedHessian(const std::shared_ptr<dotk::DOTk_Primal> & input_,
                                     const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_);
    NumericallyDifferentiatedHessian(const std::shared_ptr<dotk::DOTk_Primal> & input_,
                                     const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                     const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~NumericallyDifferentiatedHessian();

    void setForwardDifference(const dotk::Vector<Real> & input_, Real epsilon_ = 1e-6);
    void setBackwardDifference(const dotk::Vector<Real> & input_, Real epsilon_ = 1e-6);
    void setCentralDifference(const dotk::Vector<Real> & input_, Real epsilon_ = 1e-6);
    void setSecondOrderForwardDifference(const dotk::Vector<Real> & input_, Real epsilon_ = 1e-6);
    void setThirdOrderForwardDifference(const dotk::Vector<Real> & input_, Real epsilon_ = 1e-6);
    void setThirdOrderBackwardDifference(const dotk::Vector<Real> & input_, Real epsilon_ = 1e-6);

    void apply(const std::shared_ptr<dotk::Vector<Real> > & primal_,
               const std::shared_ptr<dotk::Vector<Real> > & gradient_,
               const std::shared_ptr<dotk::Vector<Real> > & vector_,
               const std::shared_ptr<dotk::Vector<Real> > & output_);
    void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
               const std::shared_ptr<dotk::Vector<Real> > & vector_,
               const std::shared_ptr<dotk::Vector<Real> > & output_);

    void setNumOtimizationItrDone(size_t itr_);
    void updateLimitedMemoryStorage(bool update_);

private:
    std::shared_ptr<dotk::DOTk_Functor> m_GradientFunctor;
    std::shared_ptr<dotk::DOTk_NumericalDifferentiation> m_NumericalDifferentiation;
};

}

#endif /* DOTK_NUMERICALLYDIFFERENTIATEDHESSIAN_HPP_ */
