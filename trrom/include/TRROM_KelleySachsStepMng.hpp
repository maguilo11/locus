/*
 * TRROM_KelleySachsStepMng.hpp
 *
 *  Created on: Apr 16, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_KELLEYSACHSSTEPMNG_HPP_
#define TRROM_KELLEYSACHSSTEPMNG_HPP_

#include "TRROM_TrustRegionStepMng.hpp"

namespace trrom
{

class Data;
class LinearOperator;
class Preconditioner;
class BoundConstraints;
class SteihaugTointSolver;
class OptimizationDataMng;
class TrustRegionNewtonIO;

template<typename ScalarType>
class Vector;

class KelleySachsStepMng : public trrom::TrustRegionStepMng
{
public:
    KelleySachsStepMng(const std::shared_ptr<trrom::Data> & data_,
                       const std::shared_ptr<trrom::LinearOperator> & linear_operator_);
    KelleySachsStepMng(const std::shared_ptr<trrom::Data> & data_,
                       const std::shared_ptr<trrom::LinearOperator> & linear_operator_,
                       const std::shared_ptr<trrom::Preconditioner> & preconditioner_);
    virtual ~KelleySachsStepMng();

    double getEta() const;
    void setEta(double input_);
    double getEpsilon() const;
    void setEpsilon(double input_);
    double getStationarityMeasure() const;
    double getMidObejectiveFunctionValue() const;
    const std::shared_ptr<trrom::Vector<double> > & getMidPrimal() const;

    bool solveSubProblem(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                         const std::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                         const std::shared_ptr<trrom::TrustRegionNewtonIO> & io_);

private:
    void bounds(const std::shared_ptr<trrom::Data> & data_);
    void initialize(const std::shared_ptr<trrom::Data> & data_);
    bool updateTrustRegionRadius(const std::shared_ptr<trrom::OptimizationDataMng> & mng_);
    void applyProjectedTrialStepToHessian(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                          const std::shared_ptr<trrom::SteihaugTointSolver> & solver_);
    double computeActualReductionLowerBound(const std::shared_ptr<trrom::OptimizationDataMng> & mng_);
    void computeActiveAndInactiveSet(const std::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                     const std::shared_ptr<trrom::SteihaugTointSolver> & solver_);

private:
    double m_Eta;
    double m_Epsilon;
    double m_StationarityMeasure;
    double m_NormInactiveGradient;
    double m_MidObjectiveFunctionValue;

    bool m_TrustRegionRadiusFlag;

    std::shared_ptr<trrom::LinearOperator> m_LinearOperator;
    std::shared_ptr<trrom::Preconditioner> m_Preconditioner;
    std::shared_ptr<trrom::BoundConstraints> m_BoundConstraint;

    std::shared_ptr<trrom::Vector<double> > m_MidPrimal;
    std::shared_ptr<trrom::Vector<double> > m_LowerBound;
    std::shared_ptr<trrom::Vector<double> > m_UpperBound;
    std::shared_ptr<trrom::Vector<double> > m_WorkVector;
    std::shared_ptr<trrom::Vector<double> > m_LowerBoundLimit;
    std::shared_ptr<trrom::Vector<double> > m_UpperBoundLimit;
    std::shared_ptr<trrom::Vector<double> > m_InactiveGradient;
    std::shared_ptr<trrom::Vector<double> > m_ProjectedTrialStep;
    std::shared_ptr<trrom::Vector<double> > m_ProjectedCauchyStep;
    std::shared_ptr<trrom::Vector<double> > m_ActiveProjectedTrialStep;
    std::shared_ptr<trrom::Vector<double> > m_InactiveProjectedTrialStep;

private:
    KelleySachsStepMng(const trrom::KelleySachsStepMng &);
    trrom::KelleySachsStepMng & operator=(const trrom::KelleySachsStepMng & rhs_);
};

}

#endif /* TRROM_KELLEYSACHSSTEPMNG_HPP_ */
