/*
 * DOTk_KelleySachsStepMng.hpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_KELLEYSACHSSTEPMNG_HPP_
#define DOTK_KELLEYSACHSSTEPMNG_HPP_

#include "DOTk_TrustRegionStepMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_Preconditioner;
class DOTk_BoundConstraints;
class DOTk_SteihaugTointSolver;
class DOTk_OptimizationDataMng;
class DOTk_SteihaugTointNewtonIO;

template<typename ScalarType>
class Vector;

class DOTk_KelleySachsStepMng: public dotk::DOTk_TrustRegionStepMng
{
public:
    DOTk_KelleySachsStepMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                            const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_);
    DOTk_KelleySachsStepMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                            const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                            const std::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_);
    virtual ~DOTk_KelleySachsStepMng();

    Real getEta() const;
    void setEta(Real input_);
    Real getEpsilon() const;
    void setEpsilon(Real input_);
    Real getStationarityMeasure() const;
    Real getMidObejectiveFunctionValue() const;
    const std::shared_ptr<dotk::Vector<Real> > & getMidPrimal() const;

    virtual void setNumOptimizationItrDone(const size_t & input_);
    virtual void solveSubProblem(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                 const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                 const std::shared_ptr<dotk::DOTk_SteihaugTointNewtonIO> & io_);

private:
    void bounds(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    bool updateTrustRegionRadius(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    void applyProjectedTrialStepToHessian(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                          const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_);
    Real computeActualReductionLowerBound(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    void computeActiveAndInactiveSet(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                     const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_);

private:
    Real m_Eta;
    Real m_Epsilon;
    Real m_StationarityMeasure;
    Real m_NormInactiveGradient;
    Real m_MidObjectiveFunctionValue;

    bool m_TrustRegionRadiusFlag;

    std::shared_ptr<dotk::DOTk_LinearOperator> m_LinearOperator;
    std::shared_ptr<dotk::DOTk_Preconditioner> m_Preconditioner;
    std::shared_ptr<dotk::DOTk_BoundConstraints> m_BoundConstraint;

    std::shared_ptr<dotk::Vector<Real> > m_MidPrimal;
    std::shared_ptr<dotk::Vector<Real> > m_LowerBound;
    std::shared_ptr<dotk::Vector<Real> > m_UpperBound;
    std::shared_ptr<dotk::Vector<Real> > m_WorkVector;
    std::shared_ptr<dotk::Vector<Real> > m_LowerBoundLimit;
    std::shared_ptr<dotk::Vector<Real> > m_UpperBoundLimit;
    std::shared_ptr<dotk::Vector<Real> > m_InactiveGradient;
    std::shared_ptr<dotk::Vector<Real> > m_ProjectedTrialStep;
    std::shared_ptr<dotk::Vector<Real> > m_ProjectedCauchyStep;
    std::shared_ptr<dotk::Vector<Real> > m_ActiveProjectedTrialStep;
    std::shared_ptr<dotk::Vector<Real> > m_InactiveProjectedTrialStep;

private:
    DOTk_KelleySachsStepMng(const dotk::DOTk_KelleySachsStepMng &);
    dotk::DOTk_KelleySachsStepMng & operator=(const dotk::DOTk_KelleySachsStepMng & rhs_);
};

}

#endif /* DOTK_KELLEYSACHSSTEPMNG_HPP_ */
