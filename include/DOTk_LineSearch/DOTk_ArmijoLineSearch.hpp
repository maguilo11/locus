/*
 * DOTk_ArmijoLineSearch.hpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ARMIJOLINESEARCH_HPP_
#define DOTK_ARMIJOLINESEARCH_HPP_

#include "DOTk_LineSearch.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_ArmijoLineSearch : public dotk::DOTk_LineSearch
{
public:
    explicit DOTk_ArmijoLineSearch(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_);
    virtual ~DOTk_ArmijoLineSearch();

    virtual Real getConstant() const;
    virtual void setConstant(Real value_);
    virtual void step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    Real m_ArmijoRuleConstant;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TrialPrimal;

private:
    // unimplemented
    DOTk_ArmijoLineSearch(const dotk::DOTk_ArmijoLineSearch &);
    DOTk_ArmijoLineSearch & operator=(const dotk::DOTk_ArmijoLineSearch & rhs_);
};

}

#endif /* DOTK_ARMIJOLINESEARCH_HPP_ */
