/*
 * DOTk_HagerZhangLineSearch.hpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_HAGERZHANGLINESEARCH_HPP_
#define DOTK_HAGERZHANGLINESEARCH_HPP_

#include <map>

#include "DOTk_LineSearch.hpp"

namespace dotk
{

template<typename Type>
class vector;

class DOTk_OptimizationDataMng;

class DOTk_HagerZhangLineSearch : public dotk::DOTk_LineSearch
{
public:
    explicit DOTk_HagerZhangLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_HagerZhangLineSearch();

    void setMaxShrinkIntervalIterations(size_t value_);
    size_t getMaxShrinkIntervalIterations() const;
    void setConstant(Real value_);
    Real getConstant() const;
    void setCurvatureConstant(Real value_);
    Real getCurvatureConstant() const;
    void setIntervalUpdateParameter(Real value_);
    Real getIntervalUpdateParameter() const;
    void setBisectionUpdateParameter(Real value_);
    Real getBisectionUpdateParameter() const;
    void setObjectiveFunctionErrorEstimateParameter(Real value_);
    Real getObjectiveFunctionErrorEstimateParameter() const;
    void setStepInterval(dotk::types::bound_t type_, Real value_);
    Real getStepInterval(dotk::types::bound_t type_);

    Real secantStep(const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                    const std::tr1::shared_ptr<dotk::vector<Real> > & primal_old_,
                    const std::tr1::shared_ptr<dotk::vector<Real> > & primal_new_,
                    const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_new_,
                    const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    void doubleSecantStep(const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                          const std::tr1::shared_ptr<dotk::vector<Real> > & primal_old_,
                          const std::tr1::shared_ptr<dotk::vector<Real> > & primal_new_,
                          const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_new_,
                          const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    void updateInterval(const Real & step_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & primal_old_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & primal_new_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_new_,
                        const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    void shrinkInterval(const Real & step_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & primal_old_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & primal_new_,
                        const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_new_,
                        const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

    virtual void step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    size_t m_MaxShrinkIntervalIterations;
    Real m_ArmijoRuleConstant;
    Real m_CurvatureConstant;
    Real m_IntervalUpdateParameter;
    Real m_BisectionUpdateParameter;
    Real m_ObjectiveFunctionErrorEstimateParameter;
    std::map<dotk::types::bound_t, Real> m_StepInterval;

private:
    // unimplemented
    DOTk_HagerZhangLineSearch(const dotk::DOTk_HagerZhangLineSearch&);
    DOTk_HagerZhangLineSearch& operator=(const dotk::DOTk_HagerZhangLineSearch& rhs_);
};

}

#endif /* DOTK_HAGERZHANGLINESEARCH_HPP_ */
