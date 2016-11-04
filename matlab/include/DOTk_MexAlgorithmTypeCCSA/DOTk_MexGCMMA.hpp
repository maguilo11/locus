/*
 * DOTk_MexGCMMA.hpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXGCMMA_HPP_
#define DOTK_MEXGCMMA_HPP_

#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexMethodCCSA.hpp"

namespace dotk
{

class DOTk_MexArrayPtr;
class DOTk_AlgorithmCCSA;

class DOTk_MexGCMMA : public dotk::DOTk_MexMethodCCSA
{
public:
    explicit DOTk_MexGCMMA(const mxArray* options_[]);
    virtual ~DOTk_MexGCMMA();

    virtual void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initialize(const mxArray* options_[]);
    void printOutputFile(dotk::DOTk_AlgorithmCCSA & algorithm_);
    void solveLinearProgrammingProblem(const mxArray* input_[], mxArray* output_[]);
    void solveNonlinearProgrammingProblem(const mxArray* input_[], mxArray* output_[]);

private:
    size_t m_MaxNumberSubProblemIterations;

    double m_SubProblemResidualTolerance;
    double m_SubProblemStagnationTolerance;

    dotk::DOTk_MexArrayPtr m_ObjectiveFunction;
    dotk::DOTk_MexArrayPtr m_EqualityConstraint;
    dotk::DOTk_MexArrayPtr m_InequalityConstraint;

private:
    DOTk_MexGCMMA(const dotk::DOTk_MexGCMMA & rhs_);
    dotk::DOTk_MexGCMMA& operator=(const dotk::DOTk_MexGCMMA & rhs_);
};

}

#endif /* DOTK_MEXGCMMA_HPP_ */
