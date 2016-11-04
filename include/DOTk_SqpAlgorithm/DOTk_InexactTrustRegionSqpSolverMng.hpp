/*
 * DOTk_InexactTrustRegionSqpSolverMng.hpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INEXACTTRUSTREGIONSQPSOLVERMNG_HPP_
#define DOTK_INEXACTTRUSTREGIONSQPSOLVERMNG_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_LinearOperator;
class DOTk_FixedCriterion;
class DOTk_LeftPreconditioner;
class DOTk_TrustRegionMngTypeELP;
class DOTk_SqpDualProblemCriterion;
class DOTk_QuasiNormalProbCriterion;
class DOTk_TangentialProblemCriterion;

template<class Type>
class vector;

class DOTk_InexactTrustRegionSqpSolverMng
{
public:
    DOTk_InexactTrustRegionSqpSolverMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                        const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);
    ~DOTk_InexactTrustRegionSqpSolverMng();

    Real getDualProbResidualNorm() const;
    Real getTangentialProbResidualNorm() const;
    Real getQuasiNormalProbResidualNorm() const;
    Real getTangentialSubProbResidualNorm() const;

    size_t getNumDualProbItrDone() const;
    size_t getNumTangentialProbItrDone() const;
    size_t getNumQuasiNormalProbItrDone() const;
    size_t getNumTangentialSubProbItrDone() const;

    void setToleranceContractionFactor(Real factor_);
    Real getToleranceContractionFactor() const;
    void setQuasiNormalProblemRelativeTolerance(Real tolerance_);
    Real getQuasiNormalProblemRelativeTolerance() const;
    void setTangentialSubProbLeftPrecProjectionTolerance(Real tolerance_);
    Real getTangentialSubProbLeftPrecProjectionTolerance() const;
    void setDualProblemTolerance(Real tolerance_);
    Real getDualTolerance() const;
    void setTangentialTolerance(Real tolerance_);
    Real getTangentialTolerance() const;
    void setTangentialToleranceContractionFactor(Real factor_);
    Real getTangentialToleranceContractionFactor() const;

    void setMaxNumDualProblemItr(size_t itr_);
    size_t getMaxNumDualProblemItr() const;
    void setMaxNumTangentialProblemItr(size_t itr_);
    size_t getMaxNumTangentialProblemItr() const;
    void setMaxNumQuasiNormalProblemItr(size_t itr_);
    size_t getMaxNumQuasiNormalProblemItr() const;
    void setMaxNumTangentialSubProblemItr(size_t itr_);
    size_t getMaxNumTangentialSubProblemItr() const;

    void setDualDotGradientTolerance(Real tolerance_);
    void setQuasiNormalProblemTrustRegionRadiusPenaltyParameter(Real parameter_);
    void setDefaultKrylovSolvers(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                 const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_);

    dotk::types::solver_stop_criterion_t solveDualProb(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);
    dotk::types::solver_stop_criterion_t solveTangentialProb(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);
    dotk::types::solver_stop_criterion_t solveQuasiNormalProb(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);
    dotk::types::solver_stop_criterion_t solveTangentialSubProb(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);

    bool adjustSolversTolerance();

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    void buildTangentialSubProblemSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                         const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_);
    void computeNormalCauchyStep(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);
    void computeScaledQuasiNormalStep(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);
    void computeScaledProjectedTangentialStep(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);

private:
    Real m_ToleranceContractionFactor;

    size_t m_MaxNumDualProblemItr;
    size_t m_MaxNumTangentialProblemItr;
    size_t m_MaxNumQuasiNormalProblemItr;
    size_t m_MaxNumTangentialSubProblemItr;

    std::tr1::shared_ptr<dotk::vector<Real> > m_DualWorkVector;
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> m_DualProbSolver;
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> m_TangentialProbSolver;
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> m_QuasiNormalProbSolver;
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> m_TangentialSubProbSolver;

    std::tr1::shared_ptr<dotk::DOTk_LinearOperator> m_AugmentedSystem;
    std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> m_TangSubProbLeftPrec;
    std::tr1::shared_ptr<dotk::DOTk_FixedCriterion> m_TangentialSubProbCriterion;
    std::tr1::shared_ptr<dotk::DOTk_SqpDualProblemCriterion> m_DualProblemCriterion;
    std::tr1::shared_ptr<dotk::DOTk_QuasiNormalProbCriterion> m_QuasiNormalProbCriterion;
    std::tr1::shared_ptr<dotk::DOTk_TangentialProblemCriterion> m_TangentialProblemCriterion;

private:
    DOTk_InexactTrustRegionSqpSolverMng(const dotk::DOTk_InexactTrustRegionSqpSolverMng &);
    dotk::DOTk_InexactTrustRegionSqpSolverMng & operator=(const dotk::DOTk_InexactTrustRegionSqpSolverMng &);
};

}

#endif /* DOTK_INEXACTTRUSTREGIONSQPSOLVERMNG_HPP_ */
