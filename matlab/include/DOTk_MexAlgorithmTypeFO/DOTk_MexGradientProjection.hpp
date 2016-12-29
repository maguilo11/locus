/*
 * DOTk_MexGradientProjection.hpp
 *
 *  Created on: Oct 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXGRADIENTPROJECTION_HPP_
#define DOTK_MEXGRADIENTPROJECTION_HPP_

#include <mex.h>
#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class GradientProjectionMethod;
class DOTk_LineSearchAlgorithmsDataMng;

class DOTk_MexGradientProjection
{
public:
    explicit DOTk_MexGradientProjection(const mxArray* options_[]);
    ~DOTk_MexGradientProjection();

    void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initialize(const mxArray* options_[]);
    void solveLinearProgrammingProblem(const mxArray* input_[], mxArray* output_[]);
    void solveNonlinearProgrammingProblem(const mxArray* input_[], mxArray* output_[]);
    void outputData(const dotk::GradientProjectionMethod & algorithm_,
                    const dotk::DOTk_LineSearchAlgorithmsDataMng & mng_,
                    mxArray* output_[]);

private:
    size_t m_MaxNumIterations;
    size_t m_MaxNumLineSearchIterations;

    double m_ObjectiveTolerance;
    double m_ProjectedGradientTolerance;
    double m_LineSearchContractionFactor;
    double m_LineSearchStagnationTolerance;

    dotk::types::problem_t m_ProblemType;
    dotk::types::line_search_t m_LineSearchMethod;

    mxArray* m_ObjectiveFunction;
    mxArray* m_EqualityConstraint;

private:
    DOTk_MexGradientProjection(const dotk::DOTk_MexGradientProjection & rhs_);
    dotk::DOTk_MexGradientProjection& operator=(const dotk::DOTk_MexGradientProjection & rhs_);
};

}

#endif /* DOTK_MEXGRADIENTPROJECTION_HPP_ */
