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
    NumericallyDifferentiatedHessian(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                     const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & aObjective);
    NumericallyDifferentiatedHessian(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                     const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & aObjective,
                                     const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & aEquality);
    virtual ~NumericallyDifferentiatedHessian();

    void setForwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon = 1e-6);
    void setBackwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon = 1e-6);
    void setCentralDifference(const dotk::Vector<Real> & aVector, Real aEpsilon = 1e-6);
    void setSecondOrderForwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon = 1e-6);
    void setThirdOrderForwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon = 1e-6);
    void setThirdOrderBackwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon = 1e-6);

    void apply(const std::shared_ptr<dotk::Vector<Real> > & aPrimal,
               const std::shared_ptr<dotk::Vector<Real> > & aGradient,
               const std::shared_ptr<dotk::Vector<Real> > & aVector,
               const std::shared_ptr<dotk::Vector<Real> > & aOutput);
    void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
               const std::shared_ptr<dotk::Vector<Real> > & aVector,
               const std::shared_ptr<dotk::Vector<Real> > & aOutput);

    void setNumOtimizationItrDone(size_t aCurrentOptimizationIteration);
    void updateLimitedMemoryStorage(bool aUpdate);

private:
    std::shared_ptr<dotk::DOTk_Functor> m_GradientFunctor;
    std::shared_ptr<dotk::DOTk_NumericalDifferentiation> m_NumericalDifferentiation;
};

}

#endif /* DOTK_NUMERICALLYDIFFERENTIATEDHESSIAN_HPP_ */
