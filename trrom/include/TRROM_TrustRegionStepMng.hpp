/*
 * TRROM_TrustRegionStepMng.hpp
 *
 *  Created on: Sep 10, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_TRUSTREGIONSTEPMNG_HPP_
#define TRROM_TRUSTREGIONSTEPMNG_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

class SteihaugTointSolver;
class OptimizationDataMng;
class SteihaugTointNewtonIO;

class TrustRegionStepMng
{
public:
    TrustRegionStepMng();
    virtual ~TrustRegionStepMng();

    void setTrustRegionRadius(double input_);
    double getTrustRegionRadius() const;
    void setTrustRegionReduction(double input_);
    double getTrustRegionReduction() const;
    void setTrustRegionExpansion(double input_);
    double getTrustRegionExpansion() const;
    void setMinTrustRegionRadius(double input_);
    double getMinTrustRegionRadius() const;
    void setMaxTrustRegionRadius(double input_);
    double getMaxTrustRegionRadius() const;

    void setActualOverPredictedReductionMidBound(double input_);
    double getActualOverPredictedReductionMidBound() const;
    void setActualOverPredictedReductionLowerBound(double input_);
    double getActualOverPredictedReductionLowerBound() const;
    void setActualOverPredictedReductionUpperBound(double input_);
    double getActualOverPredictedReductionUpperBound() const;

    void setActualReduction(double input_);
    double getActualReduction() const;
    void setPredictedReduction(double input_);
    double getPredictedReduction() const;
    void setMinCosineAngleTolerance(double tol_);
    double getMinCosineAngleTolerance() const;
    void setTrustRegionRadiusScaling(double input_);
    double getTrustRegionRadiusScaling() const;
    void setActualOverPredictedReduction(double input_);
    double getActualOverPredictedReduction() const;

    void updateObjectiveInexactnessTolerance(double predicted_reduction_);
    double getObjectiveInexactnessTolerance() const;
    void setObjectiveInexactnessToleranceConstant(double input_);
    double getObjectiveInexactnessToleranceConstant() const;

    void updateGradientInexactnessTolerance(double norm_gradient_);
    double getGradientInexactnessTolerance() const;
    void setGradientInexactnessToleranceConstant(double input_);
    double getGradientInexactnessToleranceConstant() const;

    void setNumTrustRegionSubProblemItrDone(int input_);
    void updateNumTrustRegionSubProblemItrDone();
    int getNumTrustRegionSubProblemItrDone() const;
    void setMaxNumTrustRegionSubProblemItr(int input_);
    int getMaxNumTrustRegionSubProblemItr() const;

    void setInitialTrustRegionRadiusToGradNorm(bool input_);
    bool isInitialTrustRegionRadiusSetToGradNorm() const;

    virtual bool solveSubProblem(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & mng_,
                                 const std::tr1::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                 const std::tr1::shared_ptr<trrom::SteihaugTointNewtonIO> & io_) = 0;

private:
    double m_ActualReduction;
    double m_TrustRegionRadius;
    double m_PredictedReduction;
    double m_MinTrustRegionRadius;
    double m_MaxTrustRegionRadius;
    double m_TrustRegionReduction;
    double m_TrustRegionExpansion;
    double m_MinCosineAngleTolerance;
    double m_TrustRegionRadiusScaling;
    double m_GradientInexactnessTolerance;
    double m_ObjectiveInexactnessTolerance;

    double m_ActualOverPredictedReduction;
    double m_ActualOverPredictedReductionMidBound;
    double m_ActualOverPredictedReductionLowerBound;
    double m_ActualOverPredictedReductionUpperBound;

    double m_GradientInexactnessToleranceConstant;
    double m_ObjectiveInexactnessToleranceConstant;

    bool m_InitialTrustRegionSetToGradNorm;

    int m_NumTrustRegionSubProblemItrDone;
    int m_MaxNumTrustRegionSubProblemItr;

private:
    TrustRegionStepMng(const trrom::TrustRegionStepMng &);
    trrom::TrustRegionStepMng & operator=(const trrom::TrustRegionStepMng & rhs_);
};


}

#endif /* TRROM_TRUSTREGIONSTEPMNG_HPP_ */
