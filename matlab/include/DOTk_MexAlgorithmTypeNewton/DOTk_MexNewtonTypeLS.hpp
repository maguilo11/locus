/*
 * DOTk_MexNewtonTypeLS.hpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXNEWTONTYPELS_HPP_
#define DOTK_MEXNEWTONTYPELS_HPP_

#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexAlgorithmTypeNewton.hpp"

namespace dotk
{

class DOTk_LineSearchStepMng;
class DOTk_LineSearchInexactNewton;
class DOTk_LineSearchAlgorithmsDataMng;

class DOTk_MexNewtonTypeLS : public dotk::DOTk_MexAlgorithmTypeNewton
{
public:
    explicit DOTk_MexNewtonTypeLS(const mxArray* options_[]);
    ~DOTk_MexNewtonTypeLS();

    size_t getMaxNumLineSearchItr() const;
    double getLineSearchContractionFactor() const;
    double getLineSearchStagnationTolerance() const;
    dotk::types::line_search_t getLineSearchMethod() const;

    void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initialize(const mxArray* options_[]);

    void solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);

    void setAlgorithmParameters(dotk::DOTk_LineSearchInexactNewton & algorithm_);
    void setLineSearchMethodParameters(const std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_);
    void optimize(const std::tr1::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_,
                  const mxArray* input_[],
                  mxArray* output_[]);

private:
    size_t m_MaxNumLineSearchItr;
    double m_LineSearchContractionFactor;
    double m_LineSearchStagnationTolerance;
    dotk::types::line_search_t m_LineSearchMethod;

    dotk::DOTk_MexArrayPtr m_ObjectiveFunctionOperators;
    dotk::DOTk_MexArrayPtr m_EqualityConstraintOperators;

private:
    DOTk_MexNewtonTypeLS(const dotk::DOTk_MexNewtonTypeLS & rhs_);
    dotk::DOTk_MexNewtonTypeLS& operator=(const dotk::DOTk_MexNewtonTypeLS & rhs_);
};

}

#endif /* DOTK_MEXNEWTONTYPELS_HPP_ */
