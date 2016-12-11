/*
 * DOTk_OptimalityCriteriaRoutineMng.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_InequalityConstraint.hpp"
#include "DOTk_OptimalityCriteriaDataMng.hpp"
#include "DOTk_OptimalityCriteriaRoutineMng.hpp"

namespace dotk
{

DOTk_OptimalityCriteriaRoutineMng::DOTk_OptimalityCriteriaRoutineMng
(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
 const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
 const std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > & inequality_) :
        m_Objective(objective_),
        m_Equality(equality_),
        m_Inequality(inequality_)
{
}

DOTk_OptimalityCriteriaRoutineMng::~DOTk_OptimalityCriteriaRoutineMng()
{
}

void DOTk_OptimalityCriteriaRoutineMng::solveEqualityConstraint(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_)
{
    mng_->getState().fill(0);
    mng_->getOldControl().update(1., mng_->getNewControl(), 0.);
    m_Equality->solve(mng_->getNewControl(), mng_->getState());
}

Real DOTk_OptimalityCriteriaRoutineMng::evaluateObjectiveFunction(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_)
{
    mng_->setOldObjectiveFunctionValue(mng_->getNewObjectiveFunctionValue());
    Real value = m_Objective->value(mng_->getState(), mng_->getNewControl());
    mng_->setNewObjectiveFunctionValue(value);
    return (value);
}

void DOTk_OptimalityCriteriaRoutineMng::computeObjectiveFunctionGradient(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_)
{
    mng_->getObjectiveGradient().fill(0.);
    m_Objective->partialDerivativeControl(mng_->getState(), mng_->getNewControl(), mng_->getObjectiveGradient());
    Real value = mng_->getObjectiveGradient().norm();
    mng_->setNormObjectiveFunctionGradient(value);
}

void DOTk_OptimalityCriteriaRoutineMng::computeMaxControlRelativeDifference(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_)
{
    mng_->getOldControl().update(-1., mng_->getNewControl(), 1.);
    mng_->getOldControl().abs();
    Real value = mng_->getOldControl().max();
    mng_->setMaxControlRelativeDifference(value);
}

Real DOTk_OptimalityCriteriaRoutineMng::computeInequalityConstraintResidual(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_)
{
    Real residual = m_Inequality->value(mng_->getState(), mng_->getNewControl()) - m_Inequality->bound();
    mng_->setInequalityConstraintResidual(residual);
    return (residual);
}

void DOTk_OptimalityCriteriaRoutineMng::computeInequalityConstraintGradient(std::tr1::shared_ptr<dotk::DOTk_OptimalityCriteriaDataMng> & mng_)
{
    mng_->getInequalityGradient().fill(0.);
    m_Inequality->partialDerivativeControl(mng_->getState(), mng_->getNewControl(), mng_->getInequalityGradient());
}

}
