/*
 * DOTk_SubProblemGCMMA.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SUBPROBLEMGCMMA_HPP_
#define DOTK_SUBPROBLEMGCMMA_HPP_

#include "DOTk_SubProblemCCSA.hpp"

namespace dotk
{

class DOTk_DataMngCCSA;
class DOTk_DualSolverCCSA;
class DOTk_BoundConstraints;
class DOTk_DualObjectiveFunctionMMA;

template<typename ScalarType>
class Vector;

class DOTk_SubProblemGCMMA : public dotk::DOTk_SubProblemCCSA
{
    // Globally Convergent Method of Moving Asymptote (GCMMA) Data Manager
    // Nomenclature follows: Svanberg, K. (2002). A class of globally convergent
    // optimization methods based on conservative convex separable approximations.
    // SIAM journal on optimization, 12(2), 555-573.
public:
    explicit DOTk_SubProblemGCMMA(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    DOTk_SubProblemGCMMA(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_,
                         const std::shared_ptr<dotk::DOTk_DualSolverCCSA> & dual_solver_);
    ~DOTk_SubProblemGCMMA();

    dotk::ccsa::stopping_criterion_t getDualSolverStoppingCriterion() const;

    void setDualObjectiveEpsilonParameter(Real input_);
    virtual void solve(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);

private:
    void initialize();
    void checkGlobalizationScalingParameters();

    void updateState(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    bool stoppingCriteriaSatisfied(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    void updateObjectiveGlobalizationScalingParameters(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    void updateInequalityGlobalizationScalingParameters(const std::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);

private:
    Real m_ObjectiveFunctionRho;
    Real m_ObjectiveFunctionMinRho;
    Real m_NewTrialObjectiveFunctionValue;
    Real m_OldTrialObjectiveFunctionValue;

    std::shared_ptr<dotk::Vector<Real> > m_DeltaControl;
    std::shared_ptr<dotk::Vector<Real> > m_TrialControl;
    std::shared_ptr<dotk::Vector<Real> > m_InequalityConstraintRho;
    std::shared_ptr<dotk::Vector<Real> > m_TrialFeasibilityMeasures;
    std::shared_ptr<dotk::Vector<Real> > m_TrialInequalityResiduals;
    std::shared_ptr<dotk::Vector<Real> > m_InequalityConstraintMinRho;


    std::shared_ptr<dotk::DOTk_BoundConstraints> m_Bounds;
    std::shared_ptr<dotk::DOTk_DualSolverCCSA> m_DualSolver;
    std::shared_ptr<dotk::DOTk_DualObjectiveFunctionMMA> m_DualObjectiveFunction;

private:
    DOTk_SubProblemGCMMA(const dotk::DOTk_SubProblemGCMMA &);
    dotk::DOTk_SubProblemGCMMA & operator=(const dotk::DOTk_SubProblemGCMMA & rhs_);
};

}

#endif /* DOTK_SUBPROBLEMGCMMA_HPP_ */
