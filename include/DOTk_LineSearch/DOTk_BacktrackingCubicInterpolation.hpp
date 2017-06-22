/*
 * DOTk_BacktrackingCubicInterpolation.hpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_BACKTRACKINGCUBICINTERPOLATION_HPP_
#define DOTK_BACKTRACKINGCUBICINTERPOLATION_HPP_

#include <vector>
#include "DOTk_LineSearch.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_BacktrackingCubicInterpolation : public dotk::DOTk_LineSearch
{
public:
    explicit DOTk_BacktrackingCubicInterpolation(const std::shared_ptr<dotk::Vector<Real> > & vector_);
    virtual ~DOTk_BacktrackingCubicInterpolation();

    virtual Real getConstant() const;
    virtual void setConstant(Real value_);
    virtual void step(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    void checkBacktrackingStep(std::vector<Real> & step_);
    void getBacktrackingCubicFit(const Real innr_gradient_trialStep_,
                                 const std::vector<Real> & objective_func_val_,
                                 std::vector<Real> & step_);

private:
    Real m_ArmijoRuleConstant;
    std::shared_ptr<dotk::Vector<Real> > m_TrialPrimal;

private:
    // unimplemented
    DOTk_BacktrackingCubicInterpolation(const dotk::DOTk_BacktrackingCubicInterpolation &);
    DOTk_BacktrackingCubicInterpolation & operator=(const dotk::DOTk_BacktrackingCubicInterpolation & rhs_);
};

}

#endif /* DOTK_BACKTRACKINGCUBICINTERPOLATION_HPP_ */
