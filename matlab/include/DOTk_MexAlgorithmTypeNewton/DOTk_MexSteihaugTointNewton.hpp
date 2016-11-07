/*
 * DOTk_MexSteihaugTointNewton.hpp
 *
 *  Created on: Sep 7, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXSTEIHAUGTOINTNEWTON_HPP_
#define DOTK_MEXSTEIHAUGTOINTNEWTON_HPP_

#include <tr1/memory>

#include "DOTk_Types.hpp"
#include "DOTk_MexArrayPtr.hpp"

namespace dotk
{

class DOTk_TrustRegionStepMng;
class DOTk_SteihaugTointNewton;
class DOTk_OptimizationDataMng;

class DOTk_MexSteihaugTointNewton
{
public:
    explicit DOTk_MexSteihaugTointNewton(const mxArray* options_[]);
    virtual ~DOTk_MexSteihaugTointNewton();

    size_t getMaxNumOptimizationItr() const;
    size_t getMaxNumSubProblemItr() const;

    double getGradientTolerance() const;
    double getTrialStepTolerance() const;
    double getObjectiveTolerance() const;
    double getActualReductionTolerance() const;

    double getMaxTrustRegionRadius() const;
    double getInitialTrustRegionRadius() const;
    double getTrustRegionExpansionFactor() const;
    double getTrustRegionContractionFactor() const;
    double getActualOverPredictedReductionUpperBound() const;
    double getActualOverPredictedReductionLowerBound() const;
    double getActualOverPredictedReductionMiddleBound() const;

    bool isInitialTrustRegionRadiusSetToNormGrad() const;

    void setAlgorithmParameters(dotk::DOTk_SteihaugTointNewton & algorithm_);
    void setTrustRegionStepParameters(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionStepMng> & mng_);
    void gatherOutputData(const dotk::DOTk_SteihaugTointNewton & algorithm_,
                          const dotk::DOTk_OptimizationDataMng & mng_,
                          const dotk::DOTk_TrustRegionStepMng & step_,
                          mxArray* output_[]);

    virtual void solve(const mxArray* input_[], mxArray* output_[]) = 0;

private:
    void initialize(const mxArray* options_[]);

private:
    size_t m_MaxNumOptItr;
    size_t m_MaxNumSubProblemItr;

    double m_GradientTolerance;
    double m_TrialStepTolerance;
    double m_ObjectiveTolerance;
    double m_ActualReductionTolerance;

    double m_MaxTrustRegionRadius;
    double m_InitialTrustRegionRadius;
    double m_TrustRegionExpansionFactor;
    double m_TrustRegionContractionFactor;
    double m_ActualOverPredictedReductionLowerBound;
    double m_ActualOverPredictedReductionUpperBound;
    double m_ActualOverPredictedReductionMiddleBound;

    bool m_SetInitialTrustRegionRadiusToNormGrad;

private:
    DOTk_MexSteihaugTointNewton(const dotk::DOTk_MexSteihaugTointNewton & rhs_);
    dotk::DOTk_MexSteihaugTointNewton& operator=(const dotk::DOTk_MexSteihaugTointNewton & rhs_);
};

}

#endif /* DOTK_MEXSTEIHAUGTOINTNEWTON_HPP_ */
