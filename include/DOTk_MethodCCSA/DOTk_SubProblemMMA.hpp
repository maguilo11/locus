/*
 * DOTk_SubProblemMMA.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SUBPROBLEMMMA_HPP_
#define DOTK_SUBPROBLEMMMA_HPP_

#include "DOTk_SubProblemCCSA.hpp"

namespace dotk
{

class DOTk_DataMngCCSA;
class DOTk_DualSolverCCSA;
class DOTk_BoundConstraints;
class DOTk_DualObjectiveFunctionMMA;

template<class Type>
class vector;

class DOTk_SubProblemMMA : public dotk::DOTk_SubProblemCCSA
{
public:
    explicit DOTk_SubProblemMMA(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    DOTk_SubProblemMMA(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_,
                       const std::tr1::shared_ptr<dotk::DOTk_DualSolverCCSA> & dual_solver_);
    ~DOTk_SubProblemMMA();

    dotk::ccsa::stopping_criterion_t getDualSolverStoppingCriterion() const;

    void setDualObjectiveEpsilonParameter(Real input_);
    virtual void solve(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);

private:
    Real m_ObjectiveFunctionRho;
    std::tr1::shared_ptr<dotk::vector<Real> > m_InequalityConstraintRho;

    std::tr1::shared_ptr<dotk::DOTk_BoundConstraints> m_Bounds;
    std::tr1::shared_ptr<dotk::DOTk_DualSolverCCSA> m_DualSolver;
    std::tr1::shared_ptr<dotk::DOTk_DualObjectiveFunctionMMA> m_DualObjectiveFunction;

private:
    DOTk_SubProblemMMA(const dotk::DOTk_SubProblemMMA &);
    dotk::DOTk_SubProblemMMA & operator=(const dotk::DOTk_SubProblemMMA & rhs_);
};

}

#endif /* DOTK_SUBPROBLEMMMA_HPP_ */
