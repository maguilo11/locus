/*
 * DOTk_OptimalityCriteriaDataMng.hpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OPTIMALITYCRITERIADATAMNG_HPP_
#define DOTK_OPTIMALITYCRITERIADATAMNG_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

template<typename Type>
class vector;

class DOTk_Primal;

class DOTk_OptimalityCriteriaDataMng
{
public:
    DOTk_OptimalityCriteriaDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    ~DOTk_OptimalityCriteriaDataMng();

    Real getMoveLimit() const;
    void setMoveLimit(Real value_);
    Real getInequalityDual() const;
    void setInequalityDual(Real value_);
    Real getDampingParameter() const;
    void setDampingParameter(Real value_);
    Real getGradientTolerance() const;
    void setGradientTolerance(Real value_);
    Real getBisectionTolerance() const;
    void setBisectionTolerance(Real value_);
    Real getFeasibilityTolerance() const;
    void setFeasibilityTolerance(Real value_);
    Real getOldObjectiveFunctionValue() const;
    void setOldObjectiveFunctionValue(Real value_);
    Real getNewObjectiveFunctionValue() const;
    void setNewObjectiveFunctionValue(Real value_);
    Real getControlStagnationTolerance() const;
    void setControlStagnationTolerance(Real value_);
    Real getInequalityConstraintResidual() const;
    void setInequalityConstraintResidual(Real value_);
    Real getMaxControlRelativeDifference() const;
    void setMaxControlRelativeDifference(Real value_);
    Real getNormObjectiveFunctionGradient() const;
    void setNormObjectiveFunctionGradient(Real value_);
    Real getInequalityConstraintDualLowerBound() const;
    void setInequalityConstraintDualLowerBound(Real value_);
    Real getInequalityConstraintDualUpperBound() const;
    void setInequalityConstraintDualUpperBound(Real value_);

    size_t getMaxNumOptimizationItr() const;
    void setMaxNumOptimizationItr(size_t value_);

    dotk::vector<Real> & getState() const;
    dotk::vector<Real> & getOldControl() const;
    dotk::vector<Real> & getNewControl() const;
    dotk::vector<Real> & getControlLowerBound() const;
    dotk::vector<Real> & getControlUpperBound() const;
    dotk::vector<Real> & getObjectiveGradient() const;
    dotk::vector<Real> & getInequalityGradient() const;

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    Real m_MoveLimit;
    Real m_InequalityDual;
    Real m_DampingParameter;
    Real m_GradientTolerance;
    Real m_BisectionTolerance;
    Real m_FeasibilityTolerance;
    Real m_NewObjectiveFunctionValue;
    Real m_OldObjectiveFunctionValue;
    Real m_ControlStagnationTolerance;
    Real m_InequalityConstraintResidual;
    Real m_MaxControlRelativeDifference;
    Real m_NormObjectiveFunctionGradient;
    Real m_InequalityConstraintDualLowerBound;
    Real m_InequalityConstraintDualUpperBound;

    size_t m_MaxNumOptimizationItr;

    std::tr1::shared_ptr<dotk::vector<Real> > m_State;
    std::tr1::shared_ptr<dotk::vector<Real> > m_OldControl;
    std::tr1::shared_ptr<dotk::vector<Real> > m_NewControl;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ControlLowerBound;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ControlUpperBound;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ObjectiveGradient;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InequalityGradient;

private:
    DOTk_OptimalityCriteriaDataMng(const dotk::DOTk_OptimalityCriteriaDataMng &);
    dotk::DOTk_OptimalityCriteriaDataMng & operator=(const dotk::DOTk_OptimalityCriteriaDataMng &);
};

}

#endif /* DOTK_OPTIMALITYCRITERIADATAMNG_HPP_ */
