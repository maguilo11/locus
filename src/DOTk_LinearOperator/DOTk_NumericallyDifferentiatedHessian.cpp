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

NumericallyDifferentiatedHessian::NumericallyDifferentiatedHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                                   const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_) :
        dotk::DOTk_LinearOperator(dotk::types::HESSIAN_MATRIX),
        m_GradientFunctor(new dotk::DOTk_GradientTypeULP(objective_)),
        m_NumericalDifferentiation()
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildBackwardDifferenceHessian(primal_, m_NumericalDifferentiation);
}

NumericallyDifferentiatedHessian::NumericallyDifferentiatedHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                                   const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                                                   const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_LinearOperator(dotk::types::HESSIAN_MATRIX),
        m_GradientFunctor(new dotk::DOTk_GradientTypeUNP(primal_, objective_, equality_)),
        m_NumericalDifferentiation()
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildBackwardDifferenceHessian(primal_, m_NumericalDifferentiation);
}

NumericallyDifferentiatedHessian::~NumericallyDifferentiatedHessian()
{
}

void NumericallyDifferentiatedHessian::setForwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                            Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildForwardDifferenceHessian(primal_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setBackwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                             Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildBackwardDifferenceHessian(primal_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setCentralDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                            Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildCentralDifferenceHessian(primal_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setSecondOrderForwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                                       Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildSecondOrderForwardDifferenceHessian(primal_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setThirdOrderForwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                                      Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildThirdOrderForwardDifferenceHessian(primal_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::setThirdOrderBackwardDifference(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                                       Real epsilon_)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildThirdOrderBackwardDifferenceHessian(primal_, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(epsilon_);
}

void NumericallyDifferentiatedHessian::apply(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                                             const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_,
                                             const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                             const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    output_->fill(0);
    m_NumericalDifferentiation->differentiate(m_GradientFunctor, *primal_, *vector_, *gradient_, *output_);
}

void NumericallyDifferentiatedHessian::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                             const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                             const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    output_->fill(0);
    m_NumericalDifferentiation->differentiate(m_GradientFunctor,
                                              *mng_->getNewPrimal(),
                                              *vector_,
                                              *mng_->getNewGradient(),
                                              *output_);
}

void NumericallyDifferentiatedHessian::setNumOtimizationItrDone(size_t itr_){return;}
void NumericallyDifferentiatedHessian::updateLimitedMemoryStorage(bool update_){return;}

}
