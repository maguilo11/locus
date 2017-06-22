/*
 * DOTk_DualSolverCCSA.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DUALSOLVERCCSA_HPP_
#define DOTK_DUALSOLVERCCSA_HPP_

#include <memory>

#include "DOTk_UtilsCCSA.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class DOTk_ObjectiveFunction;

class DOTk_DualSolverCCSA
{
    // Conservative Convex Separable Approximations (CCSA) dual solver parent class
public:
    explicit DOTk_DualSolverCCSA(dotk::ccsa::dual_solver_t type_);
    virtual ~DOTk_DualSolverCCSA();

    dotk::ccsa::dual_solver_t getDualSolverType() const;
    dotk::ccsa::stopping_criterion_t getStoppingCriterion() const;
    void setStoppingCriterion(dotk::ccsa::stopping_criterion_t type_);

    size_t getIterationCount() const;
    void resetIterationCount();
    void updateIterationCount();
    size_t getMaxNumIterations() const;
    void setMaxNumIterations(size_t input_);

    Real getGradientTolerance() const;
    void setGradientTolerance(Real tolerance_);
    Real getObjectiveStagnationTolerance() const;
    void setObjectiveStagnationTolerance(Real tolerance_);
    Real getTrialStepTolerance() const;
    void setTrialStepTolerance(Real tolerance_);

    size_t getLineSearchIterationCount() const;
    void resetLineSearchIterationCount();
    void updateLineSearchIterationCount();
    size_t getMaxNumLineSearchIterations() const;
    void setMaxNumLineSearchIterations(size_t iterations_);

    Real getLineSearchStepLowerBound() const;
    void setLineSearchStepLowerBound(Real input_);
    Real getLineSearchStepUpperBound() const;
    void setLineSearchStepUpperBound(Real input_);

    virtual void reset() = 0;
    virtual void solve(const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                       const std::shared_ptr<dotk::Vector<Real> > & solution_) = 0;

private:
    dotk::ccsa::dual_solver_t m_DualSolverType;
    dotk::ccsa::stopping_criterion_t m_StoppingCriterion;

    size_t m_IterationCount;
    size_t m_MaxNumIterations;
    size_t m_LineSearchIterationCount;
    size_t m_MaxNumLineSearchIterations;

    Real m_GradientTolerance;
    Real m_TrialStepTolerance;
    Real m_LineSearchStepLowerBound;
    Real m_LineSearchStepUpperBound;
    Real m_ObjectiveStagnationTolerance;

private:
    DOTk_DualSolverCCSA(const dotk::DOTk_DualSolverCCSA &);
    dotk::DOTk_DualSolverCCSA & operator=(const dotk::DOTk_DualSolverCCSA & rhs_);
};

}

#endif /* DOTK_DUALSOLVERCCSA_HPP_ */
