/*
 * DOTk_MexMMA.hpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXMMA_HPP_
#define DOTK_MEXMMA_HPP_

#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexMethodCCSA.hpp"

namespace dotk
{

class DOTk_MexArrayPtr;
class DOTk_AlgorithmCCSA;

class DOTk_MexMMA : public dotk::DOTk_MexMethodCCSA
{
public:
    explicit DOTk_MexMMA(const mxArray* options_[]);
    virtual ~DOTk_MexMMA();

    virtual void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initialize(const mxArray* options_[]);
    void printOutputFile(dotk::DOTk_AlgorithmCCSA & algorithm_);
    void solveLinearProgrammingProblem(const mxArray* input_[], mxArray* output_[]);
    void solveNonlinearProgrammingProblem(const mxArray* input_[], mxArray* output_[]);

private:
    dotk::DOTk_MexArrayPtr m_ObjectiveFunction;
    dotk::DOTk_MexArrayPtr m_EqualityConstraint;
    dotk::DOTk_MexArrayPtr m_InequalityConstraint;

private:
    DOTk_MexMMA(const dotk::DOTk_MexMMA & rhs_);
    dotk::DOTk_MexMMA& operator=(const dotk::DOTk_MexMMA & rhs_);

};

}

#endif /* DOTK_MEXMMA_HPP_ */
