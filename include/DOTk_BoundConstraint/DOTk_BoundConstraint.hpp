/*
 * DOTk_BoundConstraint.hpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_BOUNDCONSTRAINT_HPP_
#define DOTK_BOUNDCONSTRAINT_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

template<class ScalarType>
class Vector;

class DOTk_Primal;
class DOTk_LineSearch;
class DOTk_OptimizationDataMng;

class DOTk_BoundConstraint
{
public:
    DOTk_BoundConstraint(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                         dotk::types::constraint_method_t aType = dotk::types::CONSTRAINT_METHOD_DISABLED);
    virtual ~DOTk_BoundConstraint();

    void setStepSize(Real aInput);
    Real getStepSize() const;
    void setStagnationTolerance(Real aInput);
    Real getStagnationTolerance() const;
    void setContractionStep(Real aInput);
    Real getContractionStep() const;
    void setNewObjectiveFunctionValue(Real aInput);
    Real getNewObjectiveFunctionValue() const;

    void setNumFeasibleItr(size_t aInput);
    size_t getNumFeasibleItr() const;
    void setMaxNumFeasibleItr(size_t aInput);
    size_t getMaxNumFeasibleItr() const;

    void activate(bool aInput);
    bool active() const;

    void setStepType(dotk::types::bound_step_t aType);
    dotk::types::bound_step_t getStepType() const;
    dotk::types::constraint_method_t type() const;

    Real getStep(const std::shared_ptr<dotk::DOTk_LineSearch> & aStep,
                 const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);
    Real getArmijoStep(const std::shared_ptr<dotk::DOTk_LineSearch> & aStep,
                       const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);
    Real getMinReductionStep(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);

    const std::shared_ptr<dotk::Vector<Real> > & activeSet() const;
    void project(const std::shared_ptr<dotk::Vector<Real> > & aLowerBound,
                 const std::shared_ptr<dotk::Vector<Real> > & aUpperBound,
                 const std::shared_ptr<dotk::Vector<Real> > & aPrimal);
    bool isFeasible(const std::shared_ptr<dotk::Vector<Real> > & aLowerBound,
                    const std::shared_ptr<dotk::Vector<Real> > & aUpperBound,
                    const std::shared_ptr<dotk::Vector<Real> > & aPrimal);
    void pruneActive(const std::shared_ptr<dotk::Vector<Real> > & aDirection);
    void computeActiveSet(const std::shared_ptr<dotk::Vector<Real> > & aLowerBound,
                          const std::shared_ptr<dotk::Vector<Real> > & aUpperBound,
                          const std::shared_ptr<dotk::Vector<Real> > & aPrimal);
    void computeScaledTrialStep(const std::shared_ptr<dotk::DOTk_LineSearch> & aStep,
                                const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                                const std::shared_ptr<dotk::Vector<Real> > & aPrimal);

    virtual void constraint(const std::shared_ptr<dotk::DOTk_LineSearch> & aStep,
                            const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng){}

private:
    bool m_Active;
    Real m_StepSize;
    Real m_ContractionStep;
    Real m_StagnationTolerance;
    Real m_NewObjectiveFunctionValue;

    size_t m_NumFeasibleItr;
    size_t m_MaxNumFeasibleItr;
    dotk::types::bound_step_t m_StepType;
    dotk::types::constraint_method_t m_Type;

    std::shared_ptr<dotk::Vector<Real> > m_ActiveSet;

private:
    DOTk_BoundConstraint(const dotk::DOTk_BoundConstraint &);
    dotk::DOTk_BoundConstraint operator=(const dotk::DOTk_BoundConstraint &);
};

}

#endif /* DOTK_BOUNDCONSTRAINT_HPP_ */
