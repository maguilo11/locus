/*
 * DOTk_TrustRegionStepMng.hpp
 *
 *  Created on: Sep 10, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGIONSTEPMNG_HPP_
#define DOTK_TRUSTREGIONSTEPMNG_HPP_

#include <memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_SteihaugTointSolver;
class DOTk_OptimizationDataMng;
class DOTk_SteihaugTointNewtonIO;

class DOTk_TrustRegionStepMng
{
public:
    DOTk_TrustRegionStepMng();
    virtual ~DOTk_TrustRegionStepMng();

    void setTrustRegionRadius(Real input_);
    Real getTrustRegionRadius() const;
    void setTrustRegionReduction(Real input_);
    Real getTrustRegionReduction() const;
    void setTrustRegionExpansion(Real input_);
    Real getTrustRegionExpansion() const;
    void setMinTrustRegionRadius(Real input_);
    Real getMinTrustRegionRadius() const;
    void setMaxTrustRegionRadius(Real input_);
    Real getMaxTrustRegionRadius() const;

    void setActualOverPredictedReductionMidBound(Real input_);
    Real getActualOverPredictedReductionMidBound() const;
    void setActualOverPredictedReductionLowerBound(Real input_);
    Real getActualOverPredictedReductionLowerBound() const;
    void setActualOverPredictedReductionUpperBound(Real input_);
    Real getActualOverPredictedReductionUpperBound() const;

    void setActualReduction(Real input_);
    Real getActualReduction() const;
    void setPredictedReduction(Real input_);
    Real getPredictedReduction() const;
    void setMinCosineAngleTolerance(Real tol_);
    Real getMinCosineAngleTolerance() const;
    void setTrustRegionRadiusScaling(Real input_);
    Real getTrustRegionRadiusScaling() const;
    void setActualOverPredictedReduction(Real input_);
    Real getActualOverPredictedReduction() const;

    void setAdaptiveGradientInexactnessConstant(Real input_);
    Real getAdaptiveGradientInexactnessConstant() const;
    void updateAdaptiveGradientInexactnessTolerance(Real norm_current_gradient_);
    Real getAdaptiveGradientInexactnessTolerance() const;

    void setAdaptiveObjectiveInexactnessConstant(Real input_);
    Real getAdaptiveObjectiveInexactnessConstant() const;
    void updateAdaptiveObjectiveInexactnessTolerance();
    Real getAdaptiveObjectiveInexactnessTolerance() const;

    void setNumTrustRegionSubProblemItrDone(size_t input_);
    void updateNumTrustRegionSubProblemItrDone();
    size_t getNumTrustRegionSubProblemItrDone() const;
    void setMaxNumTrustRegionSubProblemItr(size_t input_);
    size_t getMaxNumTrustRegionSubProblemItr() const;

    void setInitialTrustRegionRadiusToGradNorm(bool input_);
    bool isInitialTrustRegionRadiusSetToGradNorm() const;

    virtual bool updateTrustRegionRadius();
    virtual void setNumOptimizationItrDone(const size_t & itr_) = 0;
    virtual void solveSubProblem(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                 const std::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                 const std::shared_ptr<dotk::DOTk_SteihaugTointNewtonIO> & io_) = 0;

private:
    Real m_ActualReduction;
    Real m_TrustRegionRadius;
    Real m_PredictedReduction;
    Real m_MinTrustRegionRadius;
    Real m_MaxTrustRegionRadius;
    Real m_TrustRegionReduction;
    Real m_TrustRegionExpansion;
    Real m_MinCosineAngleTolerance;
    Real m_TrustRegionRadiusScaling;

    Real m_ActualOverPredictedReduction;
    Real m_ActualOverPredictedReductionMidBound;
    Real m_ActualOverPredictedReductionLowerBound;
    Real m_ActualOverPredictedReductionUpperBound;

    Real m_AdaptiveGradientInexactnessConstant;
    Real m_AdaptiveGradientInexactnessTolerance;
    Real m_AdaptiveObjectiveInexactnessConstant;
    Real m_AdaptiveObjectiveInexactnessTolerance;

    bool m_InitialTrustRegionSetToGradNorm;

    size_t m_NumTrustRegionSubProblemItrDone;
    size_t m_MaxNumTrustRegionSubProblemItr;

private:
    DOTk_TrustRegionStepMng(const dotk::DOTk_TrustRegionStepMng &);
    dotk::DOTk_TrustRegionStepMng & operator=(const dotk::DOTk_TrustRegionStepMng & rhs_);
};


}

#endif /* DOTK_TRUSTREGIONSTEPMNG_HPP_ */
