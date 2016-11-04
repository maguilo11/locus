/*
 * DOTk_GoldsteinLineSearch.hpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_GOLDSTEINLINESEARCH_HPP_
#define DOTK_GOLDSTEINLINESEARCH_HPP_

#include "DOTk_LineSearch.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;
template<class Type>
class vector;

class DOTk_GoldsteinLineSearch : public dotk::DOTk_LineSearch
{
public:
    explicit DOTk_GoldsteinLineSearch(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_GoldsteinLineSearch();

    virtual Real getConstant() const;
    virtual void setConstant(Real value_);
    virtual void step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    Real m_GoldsteinConstant;
    std::tr1::shared_ptr<dotk::vector<Real> > m_TrialPrimal;

private:
    // unimplemented
    DOTk_GoldsteinLineSearch(const dotk::DOTk_GoldsteinLineSearch&);
    DOTk_GoldsteinLineSearch& operator=(const dotk::DOTk_GoldsteinLineSearch& rhs_);
};

}

#endif /* DOTK_GOLDSTEINLINESEARCH_HPP_ */
