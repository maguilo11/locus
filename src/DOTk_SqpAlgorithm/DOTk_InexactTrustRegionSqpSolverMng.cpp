/*
 * DOTk_InexactTrustRegionSqpSolverMng.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>
#include <cassert>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_MathUtils.hpp"
#include "DOTk_TrustRegion.hpp"
#include "DOTk_FixedCriterion.hpp"
#include "DOTk_AugmentedSystem.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_ProjectedLeftPrecCG.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_KrylovSolverFactory.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"
#include "DOTk_SqpDualProblemCriterion.hpp"
#include "DOTk_QuasiNormalProbCriterion.hpp"
#include "DOTk_AugmentedSystemPrecFactory.hpp"
#include "DOTk_TangentialProblemCriterion.hpp"
#include "DOTk_InexactTrustRegionSqpSolverMng.hpp"

namespace dotk
{

DOTk_InexactTrustRegionSqpSolverMng::DOTk_InexactTrustRegionSqpSolverMng
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
 const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & data_mng_) :
        m_ToleranceContractionFactor(1e-1),
        m_MaxNumDualProblemItr(200),
        m_MaxNumTangentialProblemItr(200),
        m_MaxNumQuasiNormalProblemItr(200),
        m_MaxNumTangentialSubProblemItr(200),
        m_DualWorkVector(data_mng_->getNewEqualityConstraintResidual()->clone()),
        m_DualProbSolver(),
        m_TangentialProbSolver(),
        m_QuasiNormalProbSolver(),
        m_TangentialSubProbSolver(),
        m_AugmentedSystem(new dotk::DOTk_AugmentedSystem),
        m_TangSubProbLeftPrec(),
        m_TangentialSubProbCriterion(new dotk::DOTk_FixedCriterion),
        m_DualProblemCriterion(new dotk::DOTk_SqpDualProblemCriterion),
        m_QuasiNormalProbCriterion(new dotk::DOTk_QuasiNormalProbCriterion),
        m_TangentialProblemCriterion(new dotk::DOTk_TangentialProblemCriterion(data_mng_->getNewPrimal()))
{
    this->initialize(primal_);
}

DOTk_InexactTrustRegionSqpSolverMng::~DOTk_InexactTrustRegionSqpSolverMng()
{
}

Real DOTk_InexactTrustRegionSqpSolverMng::getDualProbResidualNorm() const
{
    return (m_DualProbSolver->getSolverResidualNorm());
}

Real DOTk_InexactTrustRegionSqpSolverMng::getTangentialProbResidualNorm() const
{
    return (m_TangentialProbSolver->getSolverResidualNorm());
}

Real DOTk_InexactTrustRegionSqpSolverMng::getQuasiNormalProbResidualNorm() const
{
    return (m_QuasiNormalProbSolver->getSolverResidualNorm());
}

Real DOTk_InexactTrustRegionSqpSolverMng::getTangentialSubProbResidualNorm() const
{
    return (m_TangentialSubProbSolver->getSolverResidualNorm());
}

size_t DOTk_InexactTrustRegionSqpSolverMng::getNumDualProbItrDone() const
{
    return (m_DualProbSolver->getNumSolverItrDone());
}

size_t DOTk_InexactTrustRegionSqpSolverMng::getNumTangentialProbItrDone() const
{
    return (m_TangentialProbSolver->getNumSolverItrDone());
}

size_t DOTk_InexactTrustRegionSqpSolverMng::getNumQuasiNormalProbItrDone() const
{
    return (m_QuasiNormalProbSolver->getNumSolverItrDone());
}

size_t DOTk_InexactTrustRegionSqpSolverMng::getNumTangentialSubProbItrDone() const
{
    return (m_TangentialSubProbSolver->getNumSolverItrDone());
}

void DOTk_InexactTrustRegionSqpSolverMng::setToleranceContractionFactor(Real factor_)
{
    m_ToleranceContractionFactor = factor_;
}

Real DOTk_InexactTrustRegionSqpSolverMng::getToleranceContractionFactor() const
{
    return (m_ToleranceContractionFactor);
}

void DOTk_InexactTrustRegionSqpSolverMng::setQuasiNormalProblemRelativeTolerance(Real tolerance_)
{
    m_QuasiNormalProbCriterion->setRelativeTolerance(tolerance_);
}

Real DOTk_InexactTrustRegionSqpSolverMng::getQuasiNormalProblemRelativeTolerance() const
{
    return (m_QuasiNormalProbCriterion->getRelativeTolerance());
}

void DOTk_InexactTrustRegionSqpSolverMng::setQuasiNormalProblemTrustRegionRadiusPenaltyParameter(Real parameter_)
{
    m_QuasiNormalProbCriterion->setTrustRegionRadiusPenaltyParameter(parameter_);
}

void DOTk_InexactTrustRegionSqpSolverMng::setTangentialSubProbLeftPrecProjectionTolerance(Real tolerance_)
{
    m_TangSubProbLeftPrec->setParameter(dotk::types::PROJECTED_GRADIENT_TOLERANCE, tolerance_);
}

Real DOTk_InexactTrustRegionSqpSolverMng::getTangentialSubProbLeftPrecProjectionTolerance() const
{
    return (m_TangSubProbLeftPrec->getParameter(dotk::types::PROJECTED_GRADIENT_TOLERANCE));
}

void DOTk_InexactTrustRegionSqpSolverMng::setDualProblemTolerance(Real tolerance_)
{
    m_DualProblemCriterion->setDualTolerance(tolerance_);
}

Real DOTk_InexactTrustRegionSqpSolverMng::getDualTolerance() const
{
    return (m_DualProblemCriterion->getDualTolerance());
}

void DOTk_InexactTrustRegionSqpSolverMng::setMaxNumDualProblemItr(size_t itr_)
{
    m_MaxNumDualProblemItr = itr_;
}

size_t DOTk_InexactTrustRegionSqpSolverMng::getMaxNumDualProblemItr() const
{
    return (m_MaxNumDualProblemItr);
}

void DOTk_InexactTrustRegionSqpSolverMng::setMaxNumTangentialProblemItr(size_t itr_)
{
    m_MaxNumTangentialProblemItr = itr_;
}

size_t DOTk_InexactTrustRegionSqpSolverMng::getMaxNumTangentialProblemItr() const
{
    return (m_MaxNumTangentialProblemItr);
}

void DOTk_InexactTrustRegionSqpSolverMng::setMaxNumQuasiNormalProblemItr(size_t itr_)
{
    m_MaxNumQuasiNormalProblemItr = itr_;
}

size_t DOTk_InexactTrustRegionSqpSolverMng::getMaxNumQuasiNormalProblemItr() const
{
    return (m_MaxNumQuasiNormalProblemItr);
}

void DOTk_InexactTrustRegionSqpSolverMng::setMaxNumTangentialSubProblemItr(size_t itr_)
{
    m_MaxNumTangentialSubProblemItr = itr_;
}

size_t DOTk_InexactTrustRegionSqpSolverMng::getMaxNumTangentialSubProblemItr() const
{
    return (m_MaxNumTangentialSubProblemItr);
}

void DOTk_InexactTrustRegionSqpSolverMng::setDualDotGradientTolerance(Real tolerance_)
{
    m_DualProblemCriterion->setDualDotGradientTolerance(tolerance_);
}

void DOTk_InexactTrustRegionSqpSolverMng::setTangentialTolerance(Real tolerance_)
{
    m_TangentialProblemCriterion->setTangentialTolerance(tolerance_);
}

Real DOTk_InexactTrustRegionSqpSolverMng::getTangentialTolerance() const
{
    return (m_TangentialProblemCriterion->getTangentialTolerance());
}

void DOTk_InexactTrustRegionSqpSolverMng::setTangentialToleranceContractionFactor(Real factor_)
{
    m_TangentialProblemCriterion->setTangentialToleranceContractionFactor(factor_);
}

Real DOTk_InexactTrustRegionSqpSolverMng::getTangentialToleranceContractionFactor() const
{
    return (m_TangentialProblemCriterion->getTangentialToleranceContractionFactor());
}

dotk::types::solver_stop_criterion_t DOTk_InexactTrustRegionSqpSolverMng::solveDualProb
(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    Real norm_gradient = mng_->getNewGradient()->norm();
    m_DualProblemCriterion->set(dotk::types::NORM_GRADIENT, norm_gradient);

    dotk::update(1., mng_->getNewGradient(), 0., mng_->m_AugmentedSystemRightHandSide);
    dotk::scale(static_cast<Real>(-1.), mng_->m_AugmentedSystemRightHandSide);
    m_DualProbSolver->solve(mng_->m_AugmentedSystemRightHandSide, m_DualProblemCriterion, mng_);

    mng_->getDeltaDual()->update(1., *m_DualProbSolver->getDataMng()->getSolution()->dual(), 0.);
    mng_->getNewDual()->update(1., *mng_->getOldDual(), 0.);
    mng_->getNewDual()->update(static_cast<Real>(1.), *mng_->m_DeltaDual, 1.);

    return (m_DualProbSolver->getSolverStopCriterion());
}

dotk::types::solver_stop_criterion_t DOTk_InexactTrustRegionSqpSolverMng::solveTangentialProb
(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    dotk::update(1., mng_->m_ProjectedTangentialStep, 0., mng_->m_AugmentedSystemLeftHandSide);
    mng_->getAugmentedSystemLeftHandSide()->dual()->fill(static_cast<Real>(0.));
    m_TangentialProbSolver->getLinearOperator()->apply(mng_,
                                                       mng_->m_AugmentedSystemLeftHandSide,
                                                       mng_->m_TangentialStepResidual);
    dotk::update(-1., mng_->m_AugmentedSystemLeftHandSide, 1., mng_->m_TangentialStepResidual);

    Real current_trust_region_radius = mng_->getTrustRegionRadius();
    m_TangentialProblemCriterion->set(dotk::types::TRUST_REGION_RADIUS, current_trust_region_radius);

    Real norm_projected_tangential_step = mng_->m_ProjectedTangentialStep->norm();
    m_TangentialProblemCriterion->set(dotk::types::NORM_PROJECTED_TANGENTIAL_STEP, norm_projected_tangential_step);

    Real norm_tangential_step_residual = mng_->m_TangentialStepResidual->norm();
    m_TangentialProblemCriterion->set(dotk::types::NORM_TANGENTIAL_STEP_RESIDUAL, norm_tangential_step_residual);

    m_TangentialProblemCriterion->setCurrentTrialStep(mng_->getTrialStep());
    dotk::update(1., mng_->m_TangentialStepResidual, 0., mng_->m_AugmentedSystemRightHandSide);
    m_TangentialProbSolver->solve(mng_->m_AugmentedSystemRightHandSide, m_TangentialProblemCriterion, mng_);
    // Get tangential step, i.e. T_k = Wt_k + dt_k, where dt_k denotes the correction term from augmented system solution
    dotk::update(1., m_TangentialProbSolver->getDataMng()->getSolution(), 0., mng_->m_TangentialStep);
    mng_->m_TangentialStep->update(static_cast<Real>(1.0), *mng_->m_ProjectedTangentialStep, 1.);

    return (m_TangentialProbSolver->getSolverStopCriterion());
}

dotk::types::solver_stop_criterion_t DOTk_InexactTrustRegionSqpSolverMng::solveQuasiNormalProb
(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    Real current_trust_region_radius = mng_->getTrustRegionRadius();
    Real trust_region_radius_penalty_param = m_QuasiNormalProbCriterion->getTrustRegionRadiusPenaltyParameter();
    Real penalized_trust_region_radius = trust_region_radius_penalty_param * current_trust_region_radius;

    this->computeNormalCauchyStep(mng_);
    Real norm_normal_cauchy_step = mng_->m_NormalCauchyStep->norm();
    if (norm_normal_cauchy_step >= penalized_trust_region_radius)
    {
        // Cauchy step is outside the trust region, return scaled Cauchy step.
        Real scale_factor = penalized_trust_region_radius / norm_normal_cauchy_step;
        mng_->m_NormalCauchyStep->scale(scale_factor);
        mng_->m_NormalStep->update(1., *mng_->m_NormalCauchyStep, 0.);
        return (dotk::types::TRUST_REGION_VIOLATED);
    }
    // Find the Newton step, solve the augmented system to compute the Newton step
    // compute grad_x(C(x_k)) ncp_k + C(x_k)
    m_DualWorkVector->fill(0.);
    mng_->getRoutinesMng()->jacobian(mng_->getNewPrimal(), mng_->m_NormalCauchyStep, m_DualWorkVector);
    m_DualWorkVector->update(static_cast<Real>(1.), *mng_->getNewEqualityConstraintResidual(), 1.);
    Real norm_jacobian_times_normal_cauchy_step = m_DualWorkVector->norm();

    Real augmented_system_stop_tolerance =
            m_QuasiNormalProbCriterion->getRelativeTolerance() * norm_jacobian_times_normal_cauchy_step;
    m_QuasiNormalProbCriterion->setStoppingTolerance(augmented_system_stop_tolerance);
    // solve the augmented system to approximate minimum-dot solution dn_k of:
    //      min || grad_x(C(x_k))n + C(x_k) ||,  s.t. ||n|| <= zeta*trust_region_radius_k
    // assemble rhs vector, -{ncp_k; grad_x(C(x_k)) ncp_k + C(x_k)}
    dotk::update(1., mng_->m_NormalCauchyStep, 0., mng_->m_AugmentedSystemRightHandSide);
    mng_->getAugmentedSystemRightHandSide()->dual()->update(1., *m_DualWorkVector, 0.);
    mng_->m_AugmentedSystemRightHandSide->scale(static_cast<Real>(-1.));

    m_QuasiNormalProbSolver->solve(mng_->m_AugmentedSystemRightHandSide, m_QuasiNormalProbCriterion, mng_);

    // get Newton shift, dNewton = NewtonStep - NormalCauchyStep, store it in work vector
    dotk::update(1., m_QuasiNormalProbSolver->getDataMng()->getSolution(), 0., mng_->m_NormalStep);
    mng_->m_NormalStep->update(static_cast<Real>(1.), *mng_->m_NormalCauchyStep, 1.);

    Real norm_normal_step = mng_->m_NormalStep->norm();
    if (norm_normal_step > penalized_trust_region_radius)
    {
        this->computeScaledQuasiNormalStep(mng_);
    }

    return (m_QuasiNormalProbSolver->getSolverStopCriterion());
}

dotk::types::solver_stop_criterion_t DOTk_InexactTrustRegionSqpSolverMng::solveTangentialSubProb
(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    const std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & preconditioner =
            m_TangentialSubProbSolver->getDataMng()->getLeftPrec();

    Real current_trust_region_radius = mng_->getTrustRegionRadius();
    m_TangentialSubProbSolver->setTrustRegionRadius(current_trust_region_radius);
    preconditioner->setParameter(dotk::types::TRUST_REGION_RADIUS, current_trust_region_radius);

    Real norm_grad = mng_->getNewGradient()->norm();
    preconditioner->setParameter(dotk::types::NORM_GRADIENT, norm_grad);

    dotk::update(1., mng_->getNewGradient(), 0., mng_->m_TangentialSubProblemRhs);
    dotk::update(1., mng_->m_HessTimesNormalStep, 1., mng_->m_TangentialSubProblemRhs);

    m_TangentialSubProbSolver->solve(mng_->m_TangentialSubProblemRhs, m_TangentialSubProbCriterion, mng_);

    dotk::update(1., m_TangentialSubProbSolver->getDataMng()->getSolution(), 0., mng_->m_ProjectedTangentialStep);
    this->computeScaledProjectedTangentialStep(mng_);

    dotk::update(1., m_TangentialSubProbSolver->getDataMng()->getFirstSolution(), 0., mng_->m_ProjectedTangentialCauchyStep);
    dotk::update(1., m_TangentialSubProbSolver->getDataMng()->getResidual(0), 0., mng_->m_ProjectedGradient);

    return (m_TangentialSubProbSolver->getSolverStopCriterion());
}

bool DOTk_InexactTrustRegionSqpSolverMng::adjustSolversTolerance()
{
    bool tol_critical_limit_violated = false;
    Real tolerance_lower_limit = std::numeric_limits<Real>::epsilon();
    Real tolerance_contraction_factor = this->getToleranceContractionFactor();

    Real adjusted_quasi_normal_prob_tol = tolerance_contraction_factor
            * m_QuasiNormalProbCriterion->getRelativeTolerance();
    Real tangential_tol_adjusted = tolerance_contraction_factor *
            m_TangentialProblemCriterion->getTangentialTolerance();
    Real adjusted_tangential_sub_prob_projection_tol = tolerance_contraction_factor
            * m_TangSubProbLeftPrec->getParameter(dotk::types::PROJECTED_GRADIENT_TOLERANCE);
    Real dual_prob_tol_adjusted = tolerance_contraction_factor * m_DualProblemCriterion->getDualTolerance();

    if(adjusted_quasi_normal_prob_tol < tolerance_lower_limit
            || tangential_tol_adjusted < tolerance_lower_limit
            || adjusted_tangential_sub_prob_projection_tol < tolerance_lower_limit
            || dual_prob_tol_adjusted < tolerance_lower_limit)
    {
        tol_critical_limit_violated = true;
    }
    else
    {
        m_QuasiNormalProbCriterion->setRelativeTolerance(adjusted_quasi_normal_prob_tol);
        m_TangentialProblemCriterion->setTangentialTolerance(tangential_tol_adjusted);
        m_TangSubProbLeftPrec->setParameter(dotk::types::PROJECTED_GRADIENT_TOLERANCE,
                                                  adjusted_tangential_sub_prob_projection_tol);
        m_DualProblemCriterion->setDualTolerance(dual_prob_tol_adjusted);
    }

    return (tol_critical_limit_violated);

}

void DOTk_InexactTrustRegionSqpSolverMng::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    size_t max_num_itr = this->getMaxNumTangentialSubProblemItr();
    assert(this->getMaxNumDualProblemItr() > 0);

    dotk::DOTk_AugmentedSystemPrecFactory factory(max_num_itr);
    factory.buildAugmentedSystemPrecWithGmresSolver(primal_, m_TangSubProbLeftPrec);
}

void DOTk_InexactTrustRegionSqpSolverMng::computeScaledProjectedTangentialStep
(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    bool curvature_violation = m_TangentialSubProbSolver->invalidCurvatureWasDetected();
    bool trust_region_violation = m_TangentialSubProbSolver->trustRegionViolationDetected();
    if((curvature_violation == true) || (trust_region_violation == true))
    {
        mng_->invalidCurvatureDetected(curvature_violation);
        mng_->getTrustRegion()->step(mng_.get(),
                                     m_TangentialSubProbSolver->getDescentDirection(),
                                     mng_->m_ProjectedTangentialStep);
    }
}

void DOTk_InexactTrustRegionSqpSolverMng::computeScaledQuasiNormalStep
(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    Real current_trust_region_radius = mng_->getTrustRegionRadius();
    Real trust_region_radius_penalty_param = m_QuasiNormalProbCriterion->getTrustRegionRadiusPenaltyParameter();
    Real penalized_trust_region_radius = trust_region_radius_penalty_param * current_trust_region_radius;

    dotk::update(1., m_QuasiNormalProbSolver->getDataMng()->getSolution(), 0., mng_->m_NormalStep);

    Real root = mng_->computeDoglegRoot(penalized_trust_region_radius,
                                        mng_->m_NormalStep,
                                        mng_->m_NormalCauchyStep);
    mng_->m_NormalStep->scale(root);
    mng_->m_NormalStep->update(static_cast<Real>(1.0), *mng_->m_NormalCauchyStep, 1.);
}

void DOTk_InexactTrustRegionSqpSolverMng::computeNormalCauchyStep
(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & interface = mng_->getRoutinesMng();
    interface->adjointJacobian(mng_->getNewPrimal(),
                               mng_->getNewEqualityConstraintResidual(),
                               mng_->m_NormalCauchyStep);
    interface->jacobian(mng_->getNewPrimal(), mng_->m_NormalCauchyStep, m_DualWorkVector);

    Real jacobian_times_jacobianT_times_equality_constraint_dot_jacobian_times_jacobianT_times_equality_constraint =
            m_DualWorkVector->dot(*m_DualWorkVector);
    Real jacobian_times_equality_constraint_dot_jacobian_times_equality_constraint =
            mng_->m_NormalCauchyStep->dot(*mng_->m_NormalCauchyStep);
    Real scale_factor = -jacobian_times_equality_constraint_dot_jacobian_times_equality_constraint
            / jacobian_times_jacobianT_times_equality_constraint_dot_jacobian_times_jacobianT_times_equality_constraint;

    mng_->m_NormalCauchyStep->scale(scale_factor);
}

void DOTk_InexactTrustRegionSqpSolverMng::setDefaultKrylovSolvers(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                                  const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_)
{
    dotk::DOTk_KrylovSolverFactory factory;

    assert(this->getMaxNumDualProblemItr() > 0);
    factory.buildPrecGmresSolver(primal_,
                                 m_AugmentedSystem,
                                 this->getMaxNumDualProblemItr(),
                                 m_DualProbSolver);

    assert(this->getMaxNumTangentialProblemItr() > 0);
    factory.buildPrecGmresSolver(primal_,
                                 m_AugmentedSystem,
                                 this->getMaxNumTangentialProblemItr(),
                                 m_TangentialProbSolver);

    assert(this->getMaxNumTangentialProblemItr() > 0);
    factory.buildPrecGmresSolver(primal_,
                                 m_AugmentedSystem,
                                 this->getMaxNumQuasiNormalProblemItr(),
                                 m_QuasiNormalProbSolver);

    this->buildTangentialSubProblemSolver(primal_, hessian_);
}

void DOTk_InexactTrustRegionSqpSolverMng::buildTangentialSubProblemSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                                          const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_)
{
    size_t max_num_itr = this->getMaxNumTangentialSubProblemItr();
    assert(max_num_itr > 0);

    std::tr1::shared_ptr<dotk::DOTk_Primal> temp_primal(new dotk::DOTk_Primal);
    temp_primal->allocateUserDefinedDual(*primal_->dual());
    temp_primal->allocateUserDefinedControl(*primal_->control());
    if(primal_->state().use_count() > 0)
    {
        temp_primal->allocateUserDefinedState(*primal_->state());
    }
    m_TangentialSubProbSolver.reset(new dotk::DOTk_ProjectedLeftPrecCG(temp_primal, hessian_, max_num_itr));
    m_TangentialSubProbSolver->getDataMng()->setLeftPrec(m_TangSubProbLeftPrec);
}

}
