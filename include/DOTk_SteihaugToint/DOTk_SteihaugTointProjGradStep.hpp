/*
 * DOTk_SteihaugTointProjGradStep.hpp
 *
 *  Created on: Sep 10, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STEIHAUGTOINTPROJGRADSTEP_HPP_
#define DOTK_STEIHAUGTOINTPROJGRADSTEP_HPP_

#include "DOTk_TrustRegionStepMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_Preconditioner;
class DOTk_BoundConstraints;
class DOTk_ProjectedGradient;
class DOTk_SteihaugTointSolver;
class DOTk_SteihaugTointNewtonIO;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_SteihaugTointProjGradStep : public dotk::DOTk_TrustRegionStepMng
{
public:
    DOTk_SteihaugTointProjGradStep(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                   const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_);
    DOTk_SteihaugTointProjGradStep(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                   const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                   const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_);
    virtual ~DOTk_SteihaugTointProjGradStep();

    void setMaxNumProjections(size_t value_);
    size_t getMaxNumProjections() const;
    void setLineSearchContraction(Real value_);
    Real getLineSearchContraction() const;
    void setControlUpdateRoutineConstant(Real value_);
    Real getControlUpdateRoutineConstant() const;

    virtual void setNumOptimizationItrDone(const size_t & itr_);
    virtual void solveSubProblem(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                 const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                 const std::shared_ptr<dotk::DOTk_SteihaugTointNewtonIO> & io_);

private:
    void bounds(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void updateControl(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    void updateDataManager(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    void computeProjectedCauchyStep(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    size_t m_MaxNumProjections;
    Real m_LineSearchContraction;
    Real m_ControlUpdateRoutineConstant;
    Real m_PredictedReductionBasedOnCauchyStep;

    std::shared_ptr<dotk::DOTk_LinearOperator> m_LinearOperator;
    std::shared_ptr<dotk::DOTk_Preconditioner> m_Preconditioner;
    std::shared_ptr<dotk::DOTk_BoundConstraints> m_BoundConstraint;

    std::shared_ptr<dotk::Vector<Real> > m_ActiveSet;
    std::shared_ptr<dotk::Vector<Real> > m_LowerBound;
    std::shared_ptr<dotk::Vector<Real> > m_UpperBound;
    std::shared_ptr<dotk::Vector<Real> > m_WorkVector;
    std::shared_ptr<dotk::Vector<Real> > m_CurrentPrimal;
    std::shared_ptr<dotk::Vector<Real> > m_ProjectedTrialStep;
    std::shared_ptr<dotk::Vector<Real> > m_ProjectedCauchyStep;

private:
    DOTk_SteihaugTointProjGradStep(const dotk::DOTk_SteihaugTointProjGradStep &);
    dotk::DOTk_SteihaugTointProjGradStep & operator=(const dotk::DOTk_SteihaugTointProjGradStep & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTPROJGRADSTEP_HPP_ */
