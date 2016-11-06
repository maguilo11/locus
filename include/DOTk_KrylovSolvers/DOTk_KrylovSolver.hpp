/*
 * DOTk_KrylovSolver.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_KRYLOVSOLVER_HPP_
#define DOTK_KRYLOVSOLVER_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_LinearOperator;
class DOTk_KrylovSolverDataMng;
class DOTk_OptimizationDataMng;
class DOTk_KrylovSolverStoppingCriterion;

template<typename Type>
class vector;

class DOTk_KrylovSolver
{
public:
    explicit DOTk_KrylovSolver(dotk::types::krylov_solver_t type_);
    virtual ~DOTk_KrylovSolver();

    void setNumSolverItrDone(size_t itr_);
    size_t getNumSolverItrDone() const;

    void invalidCurvatureDetected(bool invalid_curvature_);
    bool invalidCurvatureWasDetected() const;
    void trustRegionViolation(bool trust_region_violation_);
    bool trustRegionViolationDetected() const;

    void setTrustRegionRadius(Real trust_region_radius_);
    Real getTrustRegionRadius() const;
    void setSolverResidualNorm(Real norm_);
    Real getSolverResidualNorm() const;
    void setInitialStoppingTolerance(Real tol_);
    Real getInitialStoppingTolerance() const;

    void setSolverType(dotk::types::krylov_solver_t type_);
    dotk::types::krylov_solver_t getSolverType() const;
    void setSolverStopCriterion(dotk::types::solver_stop_criterion_t flag_);
    dotk::types::solver_stop_criterion_t getSolverStopCriterion() const;

    bool checkCurvature(Real curvature_);
    bool checkResidualNorm(Real norm_, Real stopping_tolerance_);

    virtual void setMaxNumKrylovSolverItr(size_t itr_) = 0;
    virtual const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const = 0;
    virtual const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const = 0;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getDescentDirection() = 0;
    virtual void solve(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_,
                       const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                       const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_) = 0;

private:
    size_t m_NumSolverItrDone;
    Real m_TrustRegionRadius;
    Real m_SolverResidualNorm;
    Real m_SolverStoppingTolerance;
    bool m_InvalidCurvatureDetected;
    bool m_TrustRegionRadiusViolation;
    dotk::types::krylov_solver_t m_SolverType;
    dotk::types::solver_stop_criterion_t m_SolverStopCriterion;

private:
    DOTk_KrylovSolver(const dotk::DOTk_KrylovSolver &);
    dotk::DOTk_KrylovSolver & operator=(const dotk::DOTk_KrylovSolver &);
};

}

#endif /* DOTK_KRYLOVSOLVER_HPP_ */
