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
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal, const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_) :
        dotk::DOTk_LinearOperator(dotk::types::HESSIAN_MATRIX),
        m_GradientFunctor(std::make_shared<dotk::DOTk_GradientTypeULP>(objective_)),
        m_NumericalDifferentiation()
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildBackwardDifferenceHessian(*aPrimal->control(), m_NumericalDifferentiation);
}

NumericallyDifferentiatedHessian::NumericallyDifferentiatedHessian
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & aObjective,
 const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & aEquality) :
        dotk::DOTk_LinearOperator(dotk::types::HESSIAN_MATRIX),
        m_GradientFunctor(std::make_shared<dotk::DOTk_GradientTypeUNP>(aPrimal, aObjective, aEquality)),
        m_NumericalDifferentiation()
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildBackwardDifferenceHessian(*aPrimal->control(), m_NumericalDifferentiation);
}

NumericallyDifferentiatedHessian::~NumericallyDifferentiatedHessian()
{
}

void NumericallyDifferentiatedHessian::setForwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildForwardDifferenceHessian(aVector, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(aEpsilon);
}

void NumericallyDifferentiatedHessian::setBackwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildBackwardDifferenceHessian(aVector, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(aEpsilon);
}

void NumericallyDifferentiatedHessian::setCentralDifference(const dotk::Vector<Real> & aVector, Real aEpsilon)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildCentralDifferenceHessian(aVector, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(aEpsilon);
}

void NumericallyDifferentiatedHessian::setSecondOrderForwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildSecondOrderForwardDifferenceHessian(aVector, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(aEpsilon);
}

void NumericallyDifferentiatedHessian::setThirdOrderForwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildThirdOrderForwardDifferenceHessian(aVector, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(aEpsilon);
}

void NumericallyDifferentiatedHessian::setThirdOrderBackwardDifference(const dotk::Vector<Real> & aVector, Real aEpsilon)
{
    dotk::DOTk_NumericalDifferentiatonFactory factory;
    factory.buildThirdOrderBackwardDifferenceHessian(aVector, m_NumericalDifferentiation);
    m_NumericalDifferentiation->setEpsilon(aEpsilon);
}

void NumericallyDifferentiatedHessian::apply(const std::shared_ptr<dotk::Vector<Real> > & aPrimal,
                                             const std::shared_ptr<dotk::Vector<Real> > & aGradient,
                                             const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                             const std::shared_ptr<dotk::Vector<Real> > & aOuput)
{
    aOuput->fill(0);
    m_NumericalDifferentiation->differentiate(m_GradientFunctor, *aPrimal, *aVector, *aGradient, *aOuput);
}

void NumericallyDifferentiatedHessian::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                                             const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                             const std::shared_ptr<dotk::Vector<Real> > & aOuput)
{
    aOuput->fill(0);
    m_NumericalDifferentiation->differentiate(m_GradientFunctor,
                                              *aMng->getNewPrimal(),
                                              *aVector,
                                              *aMng->getNewGradient(),
                                              *aOuput);
}

void NumericallyDifferentiatedHessian::setNumOtimizationItrDone(size_t aCurrentOptimizationIteration)
{
    return;
}
void NumericallyDifferentiatedHessian::updateLimitedMemoryStorage(bool aUpdate)
{
    return;
}

}
