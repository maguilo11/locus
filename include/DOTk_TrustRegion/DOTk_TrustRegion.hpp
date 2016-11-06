/*
 * DOTk_TrustRegion.hpp
 *
 *  Created on: Sep 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGION_HPP_
#define DOTK_TRUSTREGION_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename Type>
class vector;

class DOTk_TrustRegion
{
public:
    DOTk_TrustRegion();
    explicit DOTk_TrustRegion(dotk::types::trustregion_t type_);
    virtual ~DOTk_TrustRegion();

    void setLineSearchIterationCount(size_t value_);
    Int getLineSearchIterationCount() const;
    void setMaxTrustRegionSubProblemIterations(Int itr_);
    size_t getMaxTrustRegionSubProblemIterations() const;
    void setNumTrustRegionSubProblemItrDone(Int itr_);
    size_t getNumTrustRegionSubProblemItrDone() const;

    void invalidCurvatureDetected(bool invalid_curvature_);
    bool isCurvatureInvalid() const;

    void setLineSearchStep(Real value_);
    Real getLineSearchStep() const;
    void setNormGradient(Real value_);
    Real getNormGradient() const;
    void setTrustRegionRadius(Real value_);
    Real getTrustRegionRadius() const;
    void setMaxTrustRegionRadius(Real value_);
    Real getMaxTrustRegionRadius() const;
    void setMinTrustRegionRadius(Real value_);
    Real getMinTrustRegionRadius() const;
    void setContractionParameter(Real value_);
    Real getContractionParameter() const;
    void setExpansionParameter(Real value_);
    Real getExpansionParameter() const;
    void setGradTimesMatrixTimesGrad(Real value_);
    Real getGradTimesMatrixTimesGrad() const;
    void setMinActualOverPredictedReductionAllowed(Real value_);
    Real getMinActualOverPredictedReductionAllowed() const;

    Real getActualReduction() const;
    Real getPredictedReduction() const;

    dotk::types::trustregion_t getTrustRegionType() const;
    void setTrustRegionType(dotk::types::trustregion_t type_);

    void computeActualReduction(Real new_objective_func_val_, Real old_objective_func_val_);
    void computePredictedReduction(const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                   const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_,
                                   const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_trial_step_);
    void computeCauchyPoint(const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & cauchy_point_);

    bool isTrustRegionStepInvalid(Real step_);
    bool acceptTrustRegionRadius(const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_);

    Real computeDoglegRoot(const Real & trust_region_radius_,
                           const std::tr1::shared_ptr<dotk::vector<Real> > & vector1_,
                           const std::tr1::shared_ptr<dotk::vector<Real> > & vector2_);
    Real computeAlternateStep(const Real & trust_region_radius_, const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);

    virtual void step(const dotk::DOTk_OptimizationDataMng * const mng_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & method_specific_required_data_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & scaled_direction_);

private:
    void shrinkTrustRegionRadius();
    void expandTrustRegionRadius(const std::tr1::shared_ptr<dotk::vector<Real> > & trial_step_);
    bool actualOverPredictedReductionViolated();

private:
    size_t mLineSearchIterationCount;
    size_t mMaxTrustRegionSubProblemIterations;
    size_t mNumTrustRegionSubProblemItrDone;

    bool mInvalidCurvatureDetected;

    Real mNormGradient;
    Real mLineSearchStep;
    Real mTrustRegionRadius;
    Real mMaxTrustRegionRadius;
    Real mMinTrustRegionRadius;
    Real mActualReduction;
    Real mPredictedReduction;
    Real mContractionParameter;
    Real mExpansionParameter;
    Real mGradTimesMatrixTimesGrad;
    Real mMinActualOverPredictedReductionAllowed;

    dotk::types::trustregion_t mTrustRegionType;

private:
    DOTk_TrustRegion(const dotk::DOTk_TrustRegion &);
    dotk::DOTk_TrustRegion operator=(const dotk::DOTk_TrustRegion &);
};

}

#endif /* DOTK_TRUSTREGION_HPP_ */
