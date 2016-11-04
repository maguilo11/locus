/*
 * DOTk_BacktrackingCubicInterpolation.hpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_BACKTRACKINGCUBICINTERPOLATION_HPP_
#define DOTK_BACKTRACKINGCUBICINTERPOLATION_HPP_

#include "DOTk_LineSearch.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;
template<class Type>
class vector;

class DOTk_BacktrackingCubicInterpolation : public dotk::DOTk_LineSearch
{
public:
    explicit DOTk_BacktrackingCubicInterpolation(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_BacktrackingCubicInterpolation();

    virtual Real getConstant() const;
    virtual void setConstant(Real value_);
    virtual void step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    void checkBacktrackingStep(std::vector<Real> & step_);
    void getBacktrackingCubicFit(const Real innr_gradient_trialStep_,
                                 const std::vector<Real> & objective_func_val_,
                                 std::vector<Real> & step_);

private:
    Real m_ArmijoRuleConstant;
    std::tr1::shared_ptr<dotk::vector<Real> > m_TrialPrimal;

private:
    // unimplemented
    DOTk_BacktrackingCubicInterpolation(const dotk::DOTk_BacktrackingCubicInterpolation &);
    DOTk_BacktrackingCubicInterpolation & operator=(const dotk::DOTk_BacktrackingCubicInterpolation & rhs_);
};

}

#endif /* DOTK_BACKTRACKINGCUBICINTERPOLATION_HPP_ */
