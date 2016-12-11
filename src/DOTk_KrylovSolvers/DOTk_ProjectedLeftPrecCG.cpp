/*
 * DOTk_ProjectedLeftPrecCG.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MathUtils.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_ProjectedLeftPrecCG.hpp"
#include "DOTk_OrthogonalProjection.hpp"
#include "DOTk_ProjLeftPrecCgDataMng.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

DOTk_ProjectedLeftPrecCG::DOTk_ProjectedLeftPrecCG(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_) :
        dotk::DOTk_KrylovSolver(dotk::types::PROJECTED_PREC_CG),
        m_InexactnessTolerance(1e-12),
        m_OrthogonalityTolerance(0.5),
        m_OrthogonalityMeasure(mng_->getMaxNumSolverItr()),
        m_OneOverNormPreconditionerTimesResidual(mng_->getMaxNumSolverItr(), 0.),
        m_DataMng(mng_),
        m_ProjectedConjugateDirection()
{
    this->initialize();
}

DOTk_ProjectedLeftPrecCG::DOTk_ProjectedLeftPrecCG(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                   const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & operator_,
                                                   size_t max_num_itr_) :
        dotk::DOTk_KrylovSolver(dotk::types::PROJECTED_PREC_CG),
        m_InexactnessTolerance(1e-12),
        m_OrthogonalityTolerance(0.5),
        m_OrthogonalityMeasure(max_num_itr_),
        m_OneOverNormPreconditionerTimesResidual(max_num_itr_, 0.),
        m_DataMng(new dotk::DOTk_ProjLeftPrecCgDataMng(primal_, operator_, max_num_itr_)),
        m_ProjectedConjugateDirection()
{
    this->initialize();
}

DOTk_ProjectedLeftPrecCG::~DOTk_ProjectedLeftPrecCG()
{
}

Real DOTk_ProjectedLeftPrecCG::getOrthogonalityTolerance() const
{
    return (m_OrthogonalityTolerance);
}

void DOTk_ProjectedLeftPrecCG::setOrthogonalityTolerance(Real tolerance_)
{
    m_OrthogonalityTolerance = tolerance_;
}

Real DOTk_ProjectedLeftPrecCG::getInexactnessTolerance() const
{
    return (m_InexactnessTolerance);
}

void DOTk_ProjectedLeftPrecCG::setInexactnessTolerance(Real tolerance_)
{
    m_InexactnessTolerance = tolerance_;
}

Real DOTk_ProjectedLeftPrecCG::getOrthogonalityMeasure(size_t row_, size_t column_) const
{
    return (m_OrthogonalityMeasure[row_][column_]);
}

void DOTk_ProjectedLeftPrecCG::setMaxNumKrylovSolverItr(size_t itr_)
{
    m_DataMng->setMaxNumSolverItr(itr_);
}

const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & DOTk_ProjectedLeftPrecCG::getDataMng() const
{
    return (m_DataMng);
}

const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & DOTk_ProjectedLeftPrecCG::getLinearOperator() const
{
    return (m_DataMng->getLinearOperator());
}

const std::tr1::shared_ptr<dotk::Vector<Real> > &
DOTk_ProjectedLeftPrecCG::getDescentDirection()
{
    return (m_ProjectedConjugateDirection);
}

bool DOTk_ProjectedLeftPrecCG::checkOrthogonalityMeasure()
{
    // compute matrix S = invD * Ymat^T * Rmat * invD - I)
    size_t krylov_subspace_itr_done = dotk::DOTk_KrylovSolver::getNumSolverItrDone();
    Real norm_preconditioner_times_residual_storage =
            m_DataMng->getLeftPrecTimesVector(krylov_subspace_itr_done)->norm();
    m_OneOverNormPreconditionerTimesResidual[krylov_subspace_itr_done] = static_cast<Real>(1.0)
            / norm_preconditioner_times_residual_storage;

    for(size_t i = 0; i <= krylov_subspace_itr_done; ++i)
    {
        for(size_t j = 0; j <= krylov_subspace_itr_done; ++j)
        {
            Real preconditioner_times_residual_dot_residual =
                    m_DataMng->getLeftPrecTimesVector(i)->dot(*m_DataMng->getResidual(j));
            m_OrthogonalityMeasure[i][j] = m_OneOverNormPreconditionerTimesResidual[i]
                    * m_OneOverNormPreconditionerTimesResidual[j] * preconditioner_times_residual_dot_residual;
            if(i == j)
            {
                m_OrthogonalityMeasure[i][j] = m_OrthogonalityMeasure[i][j] - static_cast<Real>(1.0);
            }
        }
    }

    Real orthogonality_tolerance = this->getOrthogonalityTolerance();
    Real frobenius_norm = dotk::frobeniusNorm(m_OrthogonalityMeasure);
    bool invalid_orthogonality_measure = frobenius_norm >= orthogonality_tolerance ? true: false;

    return (invalid_orthogonality_measure);
}

void DOTk_ProjectedLeftPrecCG::initialize(const std::tr1::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                                          const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    this->clear();
    size_t itr = 0;
    dotk::DOTk_KrylovSolver::trustRegionViolation(false);
    m_DataMng->getSolution()->fill(0.);
    m_DataMng->getLeftPrec()->setParameter(dotk::types::CURRENT_KRYLOV_SOLVER_ITR, itr);
    m_DataMng->getLeftPrec()->apply(mng_, rhs_vec_, m_DataMng->getLeftPrecTimesVector(itr));
    m_DataMng->getResidual(itr)->copy(*m_DataMng->getLeftPrecTimesVector(itr));
}

void DOTk_ProjectedLeftPrecCG::ppcg(const std::tr1::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                                    const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                                    const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & optimization_data_mng_)
{
    size_t itr = 0;
    this->initialize(rhs_vec_, optimization_data_mng_);
    const std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & projection = m_DataMng->getProjection();
    while(1)
    {
        this->setFirstSolution();
        if(itr > m_DataMng->getMaxNumSolverItr())
        {
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::MAX_SOLVER_ITR_REACHED);
            break;
        }
        dotk::DOTk_KrylovSolver::setNumSolverItrDone(itr);
        if(this->checkOrthogonalityMeasure() == true)
        {
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::INVALID_ORTHOGONALITY_MEASURE);
            break;
        }
        projection->apply(this, m_DataMng->getLeftPrecTimesVector(itr));
        Real norm_old_residual = m_DataMng->getResidual(itr)->norm();
        Real old_residual_dot_proj_prec_residual =
                m_DataMng->getResidual(itr)->dot(*projection->getOrthogonalVector(itr));
        m_DataMng->getLinearOperator()->apply(optimization_data_mng_,
                                              projection->getOrthogonalVector(itr),
                                              projection->getLinearOperatorTimesOrthoVector(itr));
        Real curvature = projection->getOrthogonalVector(itr)->dot(*projection->getLinearOperatorTimesOrthoVector(itr));
        if(this->checkInexactnessMeasure(curvature, norm_old_residual, old_residual_dot_proj_prec_residual) == true)
        {
            break;
        }
        Real alpha = -old_residual_dot_proj_prec_residual / curvature;
        m_DataMng->getPreviousSolution()->copy(*m_DataMng->getSolution());
        m_DataMng->getSolution()->axpy(alpha, *projection->getOrthogonalVector(itr));
        Real norm_solution = m_DataMng->getSolution()->norm();
        Real trust_region_radius = dotk::DOTk_KrylovSolver::getTrustRegionRadius();
        if(norm_solution >= trust_region_radius)
        {
            dotk::DOTk_KrylovSolver::trustRegionViolation(true);
            dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::TRUST_REGION_VIOLATED);
            break;
        }
        m_DataMng->getResidual(itr + 1)->copy(*m_DataMng->getResidual(itr));
        m_DataMng->getResidual(itr + 1)->axpy(alpha, *projection->getLinearOperatorTimesOrthoVector(itr));

        m_DataMng->getLeftPrec()->setParameter(dotk::types::CURRENT_KRYLOV_SOLVER_ITR, itr + 1);
        m_DataMng->getLeftPrec()->apply(optimization_data_mng_,
                                        m_DataMng->getResidual(itr + 1),
                                        m_DataMng->getLeftPrecTimesVector(itr + 1));

        Real norm_new_residual = m_DataMng->getLeftPrecTimesVector(itr + 1)->norm();
        dotk::DOTk_KrylovSolver::setSolverResidualNorm(norm_new_residual);
        Real stopping_tolerance = criterion_->evaluate(this, m_DataMng->getLeftPrecTimesVector(itr + 1));
        if(dotk::DOTk_KrylovSolver::checkResidualNorm(norm_new_residual, stopping_tolerance) == true)
        {
            break;
        }
        ++itr;
    }
    this->setProjectedConjugateDirection();
    if(itr == 0)
    {
        m_DataMng->getFirstSolution()->copy(*m_DataMng->getSolution());
    }
}

void DOTk_ProjectedLeftPrecCG::solve(const std::tr1::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                                     const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                                     const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_)
{
    this->ppcg(rhs_vec_, criterion_, mng_);
}

void DOTk_ProjectedLeftPrecCG::setProjectedConjugateDirection()
{
    size_t itr = dotk::DOTk_KrylovSolver::getNumSolverItrDone();
    bool curvature_violation = dotk::DOTk_KrylovSolver::invalidCurvatureWasDetected();
    bool trust_region_violation = dotk::DOTk_KrylovSolver::trustRegionViolationDetected();
    if((curvature_violation == true) || (trust_region_violation == true))
    {
        if(itr > 0)
        {
            m_ProjectedConjugateDirection->copy(*m_DataMng->getProjection()->getOrthogonalVector(itr - 1));
        }
        else
        {
            m_ProjectedConjugateDirection->copy(*m_DataMng->getProjection()->getOrthogonalVector(itr));
        }
    }
    else
    {
        m_ProjectedConjugateDirection->copy(*m_DataMng->getProjection()->getOrthogonalVector(itr));
    }
}

bool DOTk_ProjectedLeftPrecCG::checkInexactnessMeasure(Real curvature_,
                                                       Real norm_old_residual_,
                                                       Real old_residual_dot_projected_prec_residual_)
{
    bool invalid_inexactness_measure = false;
    size_t current_itr = dotk::DOTk_KrylovSolver::getNumSolverItrDone();
    const std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & projection = m_DataMng->getProjection();

    Real norm_proj_prec_res_dot_proj_prec_res = projection->getOrthogonalVector(current_itr)->norm();

    Real inexactness_tol = this->getInexactnessTolerance();
    Real inexactness_measure = inexactness_tol * norm_proj_prec_res_dot_proj_prec_res * norm_old_residual_;
    Real curvature = curvature_;
    if(dotk::DOTk_KrylovSolver::checkCurvature(curvature) == true)
    {
        bool criterion_one = old_residual_dot_projected_prec_residual_ > 0. ? true: false;
        bool criterion_two = std::abs(old_residual_dot_projected_prec_residual_) > inexactness_measure ? true: false;
        if(criterion_one && criterion_two)
        {
            projection->getOrthogonalVector(current_itr)->scale(static_cast<Real>(-1.));
        }
        invalid_inexactness_measure = true;
    }
    else if(std::abs(old_residual_dot_projected_prec_residual_) < inexactness_measure)
    {
        dotk::DOTk_KrylovSolver::setSolverStopCriterion(dotk::types::INVALID_INEXACTNESS_MEASURE);
        invalid_inexactness_measure = true;
    }

    return (invalid_inexactness_measure);
}

void DOTk_ProjectedLeftPrecCG::initialize()
{
    m_ProjectedConjugateDirection = m_DataMng->getSolution()->clone();

    size_t krylov_subspace_dim = m_DataMng->getMaxNumSolverItr();

    for(size_t i = 0; i < krylov_subspace_dim; ++i)
    {
        m_OrthogonalityMeasure[i].resize(krylov_subspace_dim, 0.);
    }
}

void DOTk_ProjectedLeftPrecCG::setFirstSolution()
{
    size_t current_itr = dotk::DOTk_KrylovSolver::getNumSolverItrDone();
    if(current_itr == 1)
    {
        m_DataMng->getFirstSolution()->copy(*m_DataMng->getSolution());
    }
}

void DOTk_ProjectedLeftPrecCG::clear()
{
    m_OneOverNormPreconditionerTimesResidual.assign(m_OneOverNormPreconditionerTimesResidual.size(), 0.);
    size_t num_rows = m_OrthogonalityMeasure.size();
    for(size_t row = 0; row < num_rows; ++row)
    {
        size_t num_columns = m_OrthogonalityMeasure[row].size();
        m_OrthogonalityMeasure[row].assign(num_columns, 0.);
    }
    m_DataMng->getProjection()->clear();
}

}
