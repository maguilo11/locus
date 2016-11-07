/*
 * DOTk_SteihaugTointSolver.hpp
 *
 *  Created on: Apr 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STEIHAUGTOINTSOLVER_HPP_
#define DOTK_STEIHAUGTOINTSOLVER_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

template<typename Type>
class vector;

class DOTk_Preconditioner;
class DOTk_LinearOperator;
class DOTk_OptimizationDataMng;

class DOTk_SteihaugTointSolver
{
public:
    DOTk_SteihaugTointSolver();
    virtual ~DOTk_SteihaugTointSolver();

    void setMaxNumItr(size_t input_);
    size_t getMaxNumItr() const;
    void setNumItrDone(size_t input_);
    size_t getNumItrDone() const;

    void setSolverTolerance(Real input_);
    Real getSolverTolerance() const;
    void setTrustRegionRadius(Real input_);
    Real getTrustRegionRadius() const;
    void setResidualNorm(Real input_);
    Real getResidualNorm() const;
    void setRelativeTolerance(Real input_);
    Real getRelativeTolerance() const;
    void setRelativeToleranceExponential(Real input_);
    Real getRelativeToleranceExponential() const;

    void setStoppingCriterion(dotk::types::solver_stop_criterion_t input_);
    dotk::types::solver_stop_criterion_t getStoppingCriterion() const;

    Real computeSteihaugTointStep(const std::tr1::shared_ptr<dotk::vector<Real> > & newton_step_,
                                  const std::tr1::shared_ptr<dotk::vector<Real> > & conjugate_dir_,
                                  const std::tr1::shared_ptr<dotk::vector<Real> > & prec_times_newton_step_,
                                  const std::tr1::shared_ptr<dotk::vector<Real> > & prec_times_conjugate_dir_);
    bool invalidCurvatureDetected(const Real & input_);
    bool toleranceSatisfied(const Real & input_);

    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getActiveSet() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getInactiveSet() const;
    virtual void solve(const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_,
                       const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                       const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_) = 0;

private:
    size_t m_MaxNumItr;
    size_t m_NumItrDone;

    Real m_Tolerance;
    Real m_ResidualNorm;
    Real m_TrustRegionRadius;
    Real m_RelativeTolerance;
    Real m_RelativeToleranceExponential;

    dotk::types::solver_stop_criterion_t m_StoppingCriterion;

private:
    DOTk_SteihaugTointSolver(const dotk::DOTk_SteihaugTointSolver &);
    dotk::DOTk_SteihaugTointSolver & operator=(const dotk::DOTk_SteihaugTointSolver & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTSOLVER_HPP_ */
