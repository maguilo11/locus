/*
 * DOTk_MexInexactNewtonTypeLS.hpp
 *
 *  Created on: Apr 25, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXINEXACTNEWTONTYPELS_HPP_
#define DOTK_MEXINEXACTNEWTONTYPELS_HPP_

#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexAlgorithmTypeNewton.hpp"

namespace dotk
{

class DOTk_LineSearchStepMng;
class DOTk_LineSearchInexactNewton;

class DOTk_MexInexactNewtonTypeLS : public dotk::DOTk_MexAlgorithmTypeNewton
{
public:
    explicit DOTk_MexInexactNewtonTypeLS(const mxArray* options_[]);
    ~DOTk_MexInexactNewtonTypeLS();

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
    void setLineSearchMethodParameters(const std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng>& step_);

private:
    size_t m_MaxNumLineSearchItr;
    double m_LineSearchContractionFactor;
    double m_LineSearchStagnationTolerance;
    dotk::types::line_search_t m_LineSearchMethod;

    dotk::DOTk_MexArrayPtr m_ObjectiveFunctionOperators;
    dotk::DOTk_MexArrayPtr m_EqualityConstraintOperators;

private:
    DOTk_MexInexactNewtonTypeLS(const dotk::DOTk_MexInexactNewtonTypeLS & rhs_);
    dotk::DOTk_MexInexactNewtonTypeLS& operator=(const dotk::DOTk_MexInexactNewtonTypeLS & rhs_);
};

}

#endif /* DOTK_MEXINEXACTNEWTONTYPELS_HPP_ */
