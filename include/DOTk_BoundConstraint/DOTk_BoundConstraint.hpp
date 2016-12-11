/*
 * DOTk_BoundConstraint.hpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_BOUNDCONSTRAINT_HPP_
#define DOTK_BOUNDCONSTRAINT_HPP_

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
    DOTk_BoundConstraint(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                         dotk::types::constraint_method_t type_ = dotk::types::CONSTRAINT_METHOD_DISABLED);
    virtual ~DOTk_BoundConstraint();

    void setStepSize(Real value_);
    Real getStepSize() const;
    void setStagnationTolerance(Real tol_);
    Real getStagnationTolerance() const;
    void setContractionStep(Real value_);
    Real getContractionStep() const;
    void setNewObjectiveFunctionValue(Real value_);
    Real getNewObjectiveFunctionValue() const;

    void setNumFeasibleItr(size_t itr_);
    size_t getNumFeasibleItr() const;
    void setMaxNumFeasibleItr(size_t itr_);
    size_t getMaxNumFeasibleItr() const;

    void activate(bool enable_);
    bool active() const;

    void setStepType(dotk::types::bound_step_t type_);
    dotk::types::bound_step_t getStepType() const;
    dotk::types::constraint_method_t type() const;

    Real getStep(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                 const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    Real getArmijoStep(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                       const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    Real getMinReductionStep(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

    const std::tr1::shared_ptr<dotk::Vector<Real> > & activeSet() const;
    void project(const std::tr1::shared_ptr<dotk::Vector<Real> > & lwr_bound_,
                 const std::tr1::shared_ptr<dotk::Vector<Real> > & upr_bound_,
                 const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_);
    bool isFeasible(const std::tr1::shared_ptr<dotk::Vector<Real> > & lower_bound_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & upper_bound_,
                    const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_);
    void pruneActive(const std::tr1::shared_ptr<dotk::Vector<Real> > & direction_);
    void computeActiveSet(const std::tr1::shared_ptr<dotk::Vector<Real> > & lower_bound_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & upper_bound_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_);
    void computeScaledTrialStep(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                                const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_);

    virtual void constraint(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                            const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_){}

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

    std::tr1::shared_ptr<dotk::Vector<Real> > m_ActiveSet;

private:
    DOTk_BoundConstraint(const dotk::DOTk_BoundConstraint &);
    dotk::DOTk_BoundConstraint operator=(const dotk::DOTk_BoundConstraint &);
};

}

#endif /* DOTK_BOUNDCONSTRAINT_HPP_ */
