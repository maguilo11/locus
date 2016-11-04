/*
 * DOTk_MexIxTrustRegionLinMore.hpp
 *
 *  Created on: Sep 7, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXIXTRUSTREGIONLINMORE_HPP_
#define DOTK_MEXIXTRUSTREGIONLINMORE_HPP_

#include "DOTk_MexSteihaugTointNewton.hpp"

namespace dotk
{

class DOTk_MexArrayPtr;
class DOTk_SteihaugTointLinMore;

class DOTk_MexIxTrustRegionLinMore : public dotk::DOTk_MexSteihaugTointNewton
{
public:
    explicit DOTk_MexIxTrustRegionLinMore(const mxArray* options_[]);
    virtual ~DOTk_MexIxTrustRegionLinMore();

    size_t getMaxNumSteihaugTointSolverItr() const;

    double getSolverRelativeTolerance() const;
    double getSolverRelativeToleranceExponential() const;

    virtual void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initializeInexactTrustRegionLinMore(const mxArray* options_[]);
    void setIxLinMoreAlgorithmParameters(dotk::DOTk_SteihaugTointLinMore & algorithm_);

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
    DOTk_MexIxTrustRegionLinMore(const dotk::DOTk_MexIxTrustRegionLinMore & rhs_);
    dotk::DOTk_MexIxTrustRegionLinMore& operator=(const dotk::DOTk_MexIxTrustRegionLinMore & rhs_);
};

}

#endif /* DOTK_MEXIXTRUSTREGIONLINMORE_HPP_ */
