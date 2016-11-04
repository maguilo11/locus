/*
 * DOTk_InexactTrustRegionSQP.hpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INEXACTTRUSTREGIONSQP_HPP_
#define DOTK_INEXACTTRUSTREGIONSQP_HPP_

#include "DOTk_SequentialQuadraticProgramming.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_TrustRegionMngTypeELP;
class DOTk_InexactTrustRegionSqpIO;
class DOTk_InexactTrustRegionSqpSolverMng;

template<class Type>
class vector;

class DOTk_InexactTrustRegionSQP: public dotk::DOTk_SequentialQuadraticProgramming
{
public:
    DOTk_InexactTrustRegionSQP(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
                               const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & data_mng_,
                               const std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng> & solver_mng_);
    virtual ~DOTk_InexactTrustRegionSQP();

    void setMaxTrustRegionRadius(Real radius_);
    void setMinTrustRegionRadius(Real radius_);
    void setInitialTrustRegionRadius(Real radius_);
    void setTrustRegionExpansionFactor(Real factor_);
    void setTrustRegionContractionFactor(Real factor_);
    void setMinActualOverPredictedReductionRatio(Real factor_);

    void setTangentialTolerance(Real tolerance_);
    void setDualProblemTolerance(Real tolerance_);
    void setToleranceContractionFactor(Real factor_);
    void setDualDotGradientTolerance(Real tolerance_);
    void setTangentialToleranceContractionFactor(Real factor_);
    void setQuasiNormalProblemRelativeTolerance(Real tolerance_);
    void setTangentialSubProbLeftPrecProjectionTolerance(Real tolerance_);
    void setQuasiNormalProblemTrustRegionRadiusPenaltyParameter(Real parameter_);

    void printDiagnosticsAndSolutionEveryItr();
    void printDiagnosticsEveryItrAndSolutionAtTheEnd();

    Real getActualOverPredictedReductionRatio() const;
    void setPredictedReductionParameter(Real parameter_);
    Real getPredictedReductionParameter() const;
    void setMeritFunctionPenaltyParameter(Real parameter_);
    Real getMeritFunctionPenaltyParameter() const;
    void setActualOverPredictedReductionTolerance(Real tolerance_);
    Real getActualOverPredictedReductionTolerance() const;
    void setMaxEffectiveTangentialOverTrialStepRatio(Real ratio_);
    Real getMaxEffectiveTangentialOverTrialStepRatio() const;

    Real getActualReduction() const;
    Real getPredictedReduction() const;

    dotk::types::solver_stop_criterion_t getDualProbExitCriterion() const;
    dotk::types::solver_stop_criterion_t getNormalProbExitCriterion() const;
    dotk::types::solver_stop_criterion_t getTangentialProbExitCriterion() const;
    dotk::types::solver_stop_criterion_t getTangentialSubProbExitCriterion() const;

    void getMin();

    void solveTrustRegionSubProblem();
    bool updateTrustRegionRadius(Real actual_over_predicted_reduction_);
    Real computeActualReduction();
    bool checkSubProblemStoppingCriteria();
    void correctTrialStep();
    void updateMeritFunctionPenaltyParameter(Real partial_predicted_reduction_);
    Real computePartialPredictedReduction();
    Real computePredictedReductionResidual();
    Real computePredictedReduction(Real partial_predicted_reduction_);

private:
    void resetState();
    void setActualReduction(Real actual_reduction_);
    void setPredictedReduction(Real predicted_reduction_);
    void setActualOverPredictedReductionRatio(Real actual_over_predicted_reduction_);
    void setDualProbExitCriterion(dotk::types::solver_stop_criterion_t criterion_);
    void setNormalProbExitCriterion(dotk::types::solver_stop_criterion_t criterion_);
    void setTangentialProbExitCriterion(dotk::types::solver_stop_criterion_t criterion_);
    void setTangentialSubProbExitCriterion(dotk::types::solver_stop_criterion_t criterion_);

private:
    Real m_ActualReduction;
    Real m_PredictedReduction;
    Real m_ActualOverPredictedReductionRatio;
    Real m_PredictedReductionParameter;
    Real m_MeritFunctionPenaltyParameter;
    Real m_ActualOverPredictedReductionTol;
    Real m_MaxEffectiveTangentialOverTrialStepRatio;

    dotk::types::solver_stop_criterion_t m_DualProbExitCriterion;
    dotk::types::solver_stop_criterion_t m_NormalProbExitCriterion;
    dotk::types::solver_stop_criterion_t m_TangentialProbExitCriterion;
    dotk::types::solver_stop_criterion_t m_TangentialSubProbExitCriterion;

    std::tr1::shared_ptr<dotk::DOTk_LinearOperator> m_Hessian;
    std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpIO> m_IO;
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> m_SqpDataMng;
    std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng> m_SqpSolverMng;

private:
    DOTk_InexactTrustRegionSQP(const dotk::DOTk_InexactTrustRegionSQP&);
    dotk::DOTk_InexactTrustRegionSQP operator=(const dotk::DOTk_InexactTrustRegionSQP&);
};

}

#endif /* DOTK_INEXACTTRUSTREGIONSQP_HPP_ */
