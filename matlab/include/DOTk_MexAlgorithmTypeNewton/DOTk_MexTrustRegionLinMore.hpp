/*
 * DOTk_MexTrustRegionLinMore.hpp
 *
 *  Created on: Apr 17, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXTRUSTREGIONLINMORE_HPP_
#define DOTK_MEXTRUSTREGIONLINMORE_HPP_

#include "DOTk_MexSteihaugTointNewton.hpp"

namespace dotk
{

class DOTk_MexArrayPtr;
class DOTk_SteihaugTointLinMore;

class DOTk_MexTrustRegionLinMore : public dotk::DOTk_MexSteihaugTointNewton
{
public:
    explicit DOTk_MexTrustRegionLinMore(const mxArray* options_[]);
    virtual ~DOTk_MexTrustRegionLinMore();

    size_t getMaxNumSteihaugTointSolverItr() const;

    double getSolverRelativeTolerance() const;
    double getSolverRelativeToleranceExponential() const;

    virtual void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initializeTrustRegionLinMore(const mxArray* options_[]);
    void setLinMoreAlgorithmParameters(dotk::DOTk_SteihaugTointLinMore & algorithm_);

    void solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeBoundNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);

private:
    dotk::types::problem_t m_ProblemType;

    size_t m_MaxNumSteihaugTointSolverItr;

    double m_SolverRelativeTolerance;
    double m_SolverRelativeToleranceExponential;

    dotk::DOTk_MexArrayPtr m_ObjectiveFunctionOperators;
    dotk::DOTk_MexArrayPtr m_EqualityConstraintOperators;

private:
    DOTk_MexTrustRegionLinMore(const dotk::DOTk_MexTrustRegionLinMore & rhs_);
    dotk::DOTk_MexTrustRegionLinMore& operator=(const dotk::DOTk_MexTrustRegionLinMore & rhs_);
};

}

#endif /* DOTK_MEXTRUSTREGIONLINMORE_HPP_ */
