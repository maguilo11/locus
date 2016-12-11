/*
 * DOTk_GoldenSectionLineSearch.hpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_GOLDENSECTIONLINESEARCH_HPP_
#define DOTK_GOLDENSECTIONLINESEARCH_HPP_

#include <vector>
#include "DOTk_LineSearch.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_GoldenSectionLineSearch : public dotk::DOTk_LineSearch
{
public:
    explicit DOTk_GoldenSectionLineSearch(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_);
    virtual ~DOTk_GoldenSectionLineSearch();

    virtual void step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    void checkGoldenSectionStep();

private:
    std::vector<Real> m_Step;
    std::vector<Real> m_ObjectiveFuncVal;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_TrialPrimal;

private:
    // unimplemented
    DOTk_GoldenSectionLineSearch(const dotk::DOTk_GoldenSectionLineSearch&);
    DOTk_GoldenSectionLineSearch& operator=(const dotk::DOTk_GoldenSectionLineSearch& rhs_);

};

}

#endif /* DOTK_GOLDENSECTIONLINESEARCH_HPP_ */
