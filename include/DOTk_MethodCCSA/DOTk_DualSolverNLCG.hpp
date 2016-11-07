/*
 * DOTk_DualSolverNLCG.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DUALSOLVERNLCG_HPP_
#define DOTK_DUALSOLVERNLCG_HPP_

#include "DOTk_DualSolverCCSA.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_BoundConstraints;
class DOTk_DataMngNonlinearCG;

template<typename Type>
class vector;
template<typename Type>
class DOTk_ObjectiveFunction;

class DOTk_DualSolverNLCG : public dotk::DOTk_DualSolverCCSA
{
    // Nonlinear Conjugate Gradient Dual Solver
public:
    explicit DOTk_DualSolverNLCG(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_DualSolverNLCG();

    dotk::types::nonlinearcg_t getNonlinearCgType() const;
    void setNonlinearCgType(dotk::types::nonlinearcg_t input_);
    void setFletcherReevesNLCG();
    void setPolakRibiereNLCG();
    void setHestenesStiefelNLCG();
    void setDaiYuanNLCG();
    void setLiuStoreyNLCG();
    void setConjugateDescentNLCG();

    Real getNewObjectiveFunctionValue() const;
    Real getOldObjectiveFunctionValue() const;

    virtual void reset();
    virtual void solve(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & solution_);
    void step(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_);

private:
    Real computeScaling();
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    bool stoppingCriteriaSatisfied();
    Real quadraticInterpolationModel(const std::vector<Real> step_values_,
                                     const std::vector<Real> objective_function_values_,
                                     const Real & initial_projected_step_dot_gradient_);

private:
    dotk::types::nonlinearcg_t m_NonlinearCgType;

    std::tr1::shared_ptr<dotk::vector<Real> > m_TrialDual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ProjectedStep;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DualLowerBound;
    std::tr1::shared_ptr<dotk::vector<Real> > m_DualUpperBound;

    std::tr1::shared_ptr<dotk::DOTk_BoundConstraints> m_Bounds;
    std::tr1::shared_ptr<dotk::DOTk_DataMngNonlinearCG> m_DataMng;

private:
    DOTk_DualSolverNLCG(const dotk::DOTk_DualSolverNLCG &);
    dotk::DOTk_DualSolverNLCG & operator=(const dotk::DOTk_DualSolverNLCG & rhs_);
};

}

#endif /* DOTK_DUALSOLVERNLCG_HPP_ */
