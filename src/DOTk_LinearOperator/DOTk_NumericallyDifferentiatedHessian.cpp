/*
 * DOTk_NumericallyDifferentiatedHessian.cpp
 *
 *  Created on: Oct 8, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_GradientTypeULP.hpp"
#include "DOTk_GradientTypeUNP.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_NumericalDifferentiation.hpp"
#include "DOTk_NumericalDifferentiatonFactory.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace dotk
{

NumericallyDifferentiatedHessian::NumericallyDifferentiatedHessian
(const std::shared_ptr<dotk::DOTk_Primal> & input_, const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_) :
        dotk::DOTk_LinearOperator(dotk::types::HESSIAN_MATRIX),
        m_GradientFunctor(new dotk::DOTk_GradientTypeULP(objective_)),
        m_NumericalDifferentiation()
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildBackwardDifferenceHessian(*input_->control(), m_NumericalDifferentiation);
}

NumericallyDifferentiatedHessian::NumericallyDifferentiatedHessian
(const std::shared_ptr<dotk::DOTk_Primal> & input_,
 const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
 const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_LinearOperator(dotk::types::HESSIAN_MATRIX),
        m_GradientFunctor(new dotk::DOTk_GradientTypeUNP(input_, objective_, equality_)),
        m_NumericalDifferentiation()
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildBackwardDifferenceHessian(*input_->control(), m_NumericalDifferentiation);
}

NumericallyDifferentiatedHessian::~NumericallyDifferentiatedHessian()
{
}

void NumericallyDifferentiatedHessian::setForwardDifference(const dotk::Vector<Real> & input_, Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildForwardDifferenceHessian(input_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setBackwardDifference(const dotk::Vector<Real> & input_, Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildBackwardDifferenceHessian(input_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setCentralDifference(const dotk::Vector<Real> & input_, Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildCentralDifferenceHessian(input_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setSecondOrderForwardDifference(const dotk::Vector<Real> & input_, Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildSecondOrderForwardDifferenceHessian(input_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setThirdOrderForwardDifference(const dotk::Vector<Real> & input_, Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildThirdOrderForwardDifferenceHessian(input_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setThirdOrderBackwardDifference(const dotk::Vector<Real> & input_, Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildThirdOrderBackwardDifferenceHessian(input_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::apply(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                                             const std::shared_ptr<dotk::Vector<Real> > & gradient_,
                                             const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                             const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    output_->fill(0);
    m_NumericalDifferentiation->differentiate(m_GradientFunctor, *primal_, *vector_, *gradient_, *output_);
}

void NumericallyDifferentiatedHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                             const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                             const std::shared_ptr<dotk::Vector<Real> > & output_)
{
    output_->fill(0);
    m_NumericalDifferentiation->differentiate(m_GradientFunctor,
                                              *mng_->getNewPrimal(),
                                              *vector_,
                                              *mng_->getNewGradient(),
                                              *output_);
}

void NumericallyDifferentiatedHessian::setNumOtimizationItrDone(size_t itr_)
{
    return;
}
void NumericallyDifferentiatedHessian::updateLimitedMemoryStorage(bool update_)
{
    return;
}

}
