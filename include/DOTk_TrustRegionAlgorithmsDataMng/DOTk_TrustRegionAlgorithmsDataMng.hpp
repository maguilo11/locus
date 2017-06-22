/*
 * DOTk_TrustRegionAlgorithmsDataMng.hpp
 *
 *  Created on: Nov 26, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGIONALGORITHMSDATAMNG_HPP_
#define DOTK_TRUSTREGIONALGORITHMSDATAMNG_HPP_

#include <sstream>
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_TrustRegion;
class DOTk_AssemblyManager;
class DOTk_FirstOrderOperator;
class DOTk_SecondOrderOperator;

template<typename ScalarType>
class Vector;

class DOTk_TrustRegionAlgorithmsDataMng: public dotk::DOTk_OptimizationDataMng
{
public:
    explicit DOTk_TrustRegionAlgorithmsDataMng(const std::shared_ptr<dotk::DOTk_Primal> & variable_);
    virtual ~DOTk_TrustRegionAlgorithmsDataMng();

    size_t getObjectiveFuncEvalCounter() const;
    size_t getHessianEvaluationCounter() const;
    size_t getGradientEvaluationCounter() const;
    size_t getAdjointHessianEvaluationCounter() const;
    size_t getEqualityConstraintEvaluationCounter() const;
    size_t getAdjointInverseJacobianWrtStateCounter() const;

    void setTrustRegionRadius(Real radius_);
    Real getTrustRegionRadius() const;
    void setMinTrustRegionRadius(Real radius_);
    Real getMinTrustRegionRadius() const;
    void setMaxTrustRegionRadius(Real radius_);
    Real getMaxTrustRegionRadius() const;
    void setTrustRegionExpansionParameter(Real parameter_);
    Real getTrustRegionExpansionParameter() const;
    void setTrustRegionContractionParameter(Real parameter_);
    Real getTrustRegionContractionParameter() const;
    void setMinActualOverPredictedReductionAllowed(Real parameter_);
    Real getMinActualOverPredictedReductionAllowed() const;

    void setMaxTrustRegionSubProblemIterations(size_t itr_);
    size_t getMaxTrustRegionSubProblemIterations() const;

    void invalidCurvatureDetected(bool invalid_curvature_detected_);

    const std::shared_ptr<dotk::DOTk_TrustRegion> & getTrustRegion() const;
    Real computeDoglegRoot(Real trust_region_radius_,
                           const std::shared_ptr<dotk::Vector<Real> > & vector1_,
                           const std::shared_ptr<dotk::Vector<Real> > & vector2_);

    void setUserDefinedGradient();
    void checkTrustRegionPtr(std::ostringstream & msg_);
    void setCauchyTrustRegionMethod(Real trust_region_radius_ = 1e4);
    void setDoglegTrustRegionMethod(Real trust_region_radius_ = 1e4);
    void setDoubleDoglegTrustRegionMethod(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                          Real trust_region_radius_ = 1e4);

    virtual void updateState(const Real new_objective_function_value_,
                             const std::shared_ptr<dotk::Vector<Real> > & new_primal_);
    virtual void computeScaledInexactNewtonStep(const bool invalid_curvature_detected_,
                                                const std::shared_ptr<dotk::Vector<Real> > & descent_direction_);

    virtual Real evaluateObjective();
    virtual Real evaluateObjective(const std::shared_ptr<dotk::Vector<Real> > & input_);
    virtual void computeGradient();
    virtual void computeGradient(const std::shared_ptr<dotk::Vector<Real> > & input_,
                                 const std::shared_ptr<dotk::Vector<Real> > & gradient_);
    virtual const std::shared_ptr<dotk::DOTk_AssemblyManager> & getRoutinesMng() const;

protected:
    std::shared_ptr<dotk::DOTk_AssemblyManager> m_RoutinesMng;
    std::shared_ptr<dotk::DOTk_FirstOrderOperator> m_FirstOrderOperator;

private:
    std::shared_ptr<dotk::DOTk_TrustRegion> m_TrustRegion;

private:
    // unimplemented
    DOTk_TrustRegionAlgorithmsDataMng(const dotk::DOTk_TrustRegionAlgorithmsDataMng &);
    dotk::DOTk_TrustRegionAlgorithmsDataMng operator=(const dotk::DOTk_TrustRegionAlgorithmsDataMng &);
};

}

#endif /* DOTK_TRUSTREGIONALGORITHMSDATAMNG_HPP_ */
