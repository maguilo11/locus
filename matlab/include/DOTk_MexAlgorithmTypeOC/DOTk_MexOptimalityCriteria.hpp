/*
 * DOTk_MexOptimalityCriteria.hpp
 *
 *  Created on: Jun 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXOPTIMALITYCRITERIA_HPP_
#define DOTK_MEXOPTIMALITYCRITERIA_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"
#include "DOTk_MexArrayPtr.hpp"

namespace dotk
{

class DOTk_OptimalityCriteria;

template<typename Type>
class vector;

class DOTk_MexOptimalityCriteria
{
public:
    explicit DOTk_MexOptimalityCriteria(const mxArray* options_[]);
    ~DOTk_MexOptimalityCriteria();

    size_t getMaxNumAlgorithmItr() const;
    double getMoveLimit() const;
    double getDualLowerBound() const;
    double getDualUpperBound() const;
    double getDampingParameter() const;
    double getGradientTolerance() const;
    double getBisectionTolerance() const;
    double getFeasibilityTolerance() const;
    double getObjectiveFunctionTolerance() const;
    double getControlStagnationTolerance() const;
    dotk::types::problem_t getProblemType() const;

    void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initialize(const mxArray* options_[]);
    void printOutputFile(dotk::DOTk_OptimalityCriteria & algorithm_);
    void gatherOutputData(dotk::DOTk_OptimalityCriteria & algorithm_,
                          dotk::vector<double> & primal_,
                          mxArray* output_[]);
    void setAlgorithmParameters(const mxArray* options_, dotk::DOTk_OptimalityCriteria & algorithm_);

private:
    size_t m_MaxNumAlgorithmItr;
    double m_MoveLimit;
    double m_DualLowerBound;
    double m_DualUpperBound;
    double m_DampingParameter;
    double m_GradientTolerance;
    double m_BisectionTolerance;
    double m_OptimalityTolerance;
    double m_FeasibilityTolerance;
    double m_ControlStagnationTolerance;
    dotk::types::problem_t m_ProblemType;

    dotk::DOTk_MexArrayPtr m_ObjectiveFunctionOperators;
    dotk::DOTk_MexArrayPtr m_EqualityConstraintOperators;
    dotk::DOTk_MexArrayPtr m_InequalityConstraintOperators;

private:
    DOTk_MexOptimalityCriteria(const dotk::DOTk_MexOptimalityCriteria & rhs_);
    dotk::DOTk_MexOptimalityCriteria& operator=(const dotk::DOTk_MexOptimalityCriteria & rhs_);
};

}

#endif /* DOTK_MEXOPTIMALITYCRITERIA_HPP_ */