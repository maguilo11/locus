
/*
 * DOTk_SubProblemCCSA.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SUBPROBLEMCCSA_HPP_
#define DOTK_SUBPROBLEMCCSA_HPP_

#include <memory>
#include "DOTk_UtilsCCSA.hpp"

namespace dotk
{

class DOTk_DataMngCCSA;

class DOTk_SubProblemCCSA
{
    // Conservative Convex Separable Approximations (CCSA) sub-problem parent class
public:
    explicit DOTk_SubProblemCCSA(dotk::ccsa::subproblem_t type_);
    virtual ~DOTk_SubProblemCCSA();

    dotk::ccsa::subproblem_t type() const;
    dotk::ccsa::stopping_criterion_t getStoppingCriterion() const;
    void setStoppingCriterion(dotk::ccsa::stopping_criterion_t type_);

    size_t getIterationCount() const;
    void resetIterationCount();
    void updateIterationCount();
    size_t getMaxNumIterations() const;
    void setMaxNumIterations(size_t input_);

    Real getResidualTolerance() const;
    void setResidualTolerance(Real tolerance_);
    Real getStagnationTolerance() const;
    void setStagnationTolerance(Real tolerance_);
    Real getDualObjectiveTrialControlBoundScaling() const;
    void setDualObjectiveTrialControlBoundScaling(Real input_);

    virtual void setDualObjectiveEpsilonParameter(Real input_) = 0;
    virtual void solve(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_) = 0;

private:
    dotk::ccsa::subproblem_t m_Type;
    dotk::ccsa::stopping_criterion_t m_StoppingCriterion;

    size_t m_IterationCount;
    size_t m_MaxNumIterations;

    Real m_ResidualTolerance;
    Real m_StagnationTolerance;
    Real m_DualObjectiveTrialControlBoundScaling;

private:
    DOTk_SubProblemCCSA(const dotk::DOTk_SubProblemCCSA &);
    dotk::DOTk_SubProblemCCSA & operator=(const dotk::DOTk_SubProblemCCSA & rhs_);
};

}

#endif /* DOTK_SUBPROBLEMCCSA_HPP_ */
