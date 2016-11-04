/*
 * DOTk_SteihaugTointNewton.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STEIHAUGTOINTNEWTON_HPP_
#define DOTK_STEIHAUGTOINTNEWTON_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_SteihaugTointNewton
{
public:
    DOTk_SteihaugTointNewton();
    virtual ~DOTk_SteihaugTointNewton();

    void setGradientTolerance(Real input_);
    Real getGradientTolerance() const;
    void setTrialStepTolerance(Real input_);
    Real getTrialStepTolerance() const;
    void setObjectiveTolerance(Real input_);
    Real getObjectiveTolerance() const;
    void setActualReductionTolerance(Real input_);
    Real getActualReductionTolerance() const;

    void setNumOptimizationItrDone(size_t input_);
    size_t getNumOptimizationItrDone() const;
    void setMaxNumOptimizationItr(size_t input_);
    size_t getMaxNumOptimizationItr() const;

    void setStoppingCriterion(dotk::types::stop_criterion_t input_);
    dotk::types::stop_criterion_t getStoppingCriterion() const;

    virtual void getMin() = 0;
    virtual void updateNumOptimizationItrDone(const size_t & input_) = 0;

private:
    Real m_GradientTolerance;
    Real m_TrialStepTolerance;
    Real m_ObjectiveTolerance;
    Real m_ActualReductionTolerance;

    size_t m_MaxNumOptimizationItr;
    size_t m_NumOptimizationItrDone;

    dotk::types::stop_criterion_t m_StoppingCriterion;


private:
    DOTk_SteihaugTointNewton(const dotk::DOTk_SteihaugTointNewton &);
    dotk::DOTk_SteihaugTointNewton & operator=(const dotk::DOTk_SteihaugTointNewton & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTNEWTON_HPP_ */
