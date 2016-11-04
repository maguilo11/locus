/*
 * DOTk_LineSearch.hpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCH_HPP_
#define DOTK_LINESEARCH_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

class DOTk_LineSearch
{
public:
    explicit DOTk_LineSearch(dotk::types::line_search_t type_ = dotk::types::LINE_SEARCH_DISABLED);
    virtual ~DOTk_LineSearch();

    size_t getMaxNumLineSearchItr() const;
    void setMaxNumLineSearchItr(size_t itr_);
    size_t getNumLineSearchItrDone() const;
    void setNumLineSearchItrDone(size_t itr_);
    Real getStepSize() const;
    void setStepSize(Real value_);
    virtual Real getConstant() const;
    virtual void setConstant(Real value_);
    Real getStepStagnationTol() const;
    void setStepStagnationTol(Real tol_);
    Real getContractionFactor() const;
    void setContractionFactor(Real value_);
    void setOldObjectiveFunctionValue(Real value_);
    Real getOldObjectiveFunctionValue() const;
    void setNewObjectiveFunctionValue(Real value_);
    Real getNewObjectiveFunctionValue() const;

    dotk::types::line_search_t type() const;

    virtual void step(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_) = 0;

private:
    size_t m_MaxNumLineSearchItr;
    size_t m_NumLineSearchItrDone;

    Real m_StepSize;
    Real m_ContractionFactor;
    Real m_StepStagnationTol;
    Real m_NewObjectiveFunctionValue;
    Real m_OldObjectiveFunctionValue;

    dotk::types::line_search_t m_Type;

private:
    DOTk_LineSearch(const dotk::DOTk_LineSearch&);
    dotk::DOTk_LineSearch & operator=(const dotk::DOTk_LineSearch & rhs_);
};

}

#endif /* DOTK_LINESEARCH_HPP_ */
