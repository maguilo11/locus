/*
 * DOTk_MexNewtonTypeTR.hpp
 *
 *  Created on: Apr 24, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXNEWTONTYPETR_HPP_
#define DOTK_MEXNEWTONTYPETR_HPP_

#include <tr1/memory>

#include "DOTk_MexAlgorithmTypeNewton.hpp"

namespace dotk
{

class DOTk_TrustRegionInexactNewton;
class DOTk_TrustRegionAlgorithmsDataMng;

class DOTk_MexNewtonTypeTR : public dotk::DOTk_MexAlgorithmTypeNewton
{
public:
    explicit DOTk_MexNewtonTypeTR(const mxArray* options_[]);
    ~DOTk_MexNewtonTypeTR();

    size_t getMaxNumTrustRegionSubProblemItr() const;

    double getMaxTrustRegionRadius() const;
    double getMinTrustRegionRadius() const;
    double getInitialTrustRegionRadius() const;
    double getTrustRegionExpansionFactor() const;
    double getTrustRegionContractionFactor() const;
    double getMinActualOverPredictedReductionRatio() const;

    void solve(const mxArray* input_[], mxArray* output_[]);

private:
    void clear();
    void initialize(const mxArray* options_[]);

    void solveTypeLinearProgramming(const mxArray* input_[], mxArray* output_[]);
    void solveTypeNonlinearProgramming(const mxArray* input_[], mxArray* output_[]);

    void setAlgorithmParameters(dotk::DOTk_TrustRegionInexactNewton & algorithm_);
    void setTrustRegionMethodParameters(const std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_);

private:
    size_t m_MaxNumTrustRegionSubProblemItr;

    double m_MaxTrustRegionRadius;
    double m_MinTrustRegionRadius;
    double m_InitialTrustRegionRadius;
    double m_TrustRegionExpansionFactor;
    double m_TrustRegionContractionFactor;
    double m_MinActualOverPredictedReductionRatio;

    mxArray* m_ObjectiveFunction;
    mxArray* m_EqualityConstraint;

private:
    DOTk_MexNewtonTypeTR(const dotk::DOTk_MexNewtonTypeTR & rhs_);
    dotk::DOTk_MexNewtonTypeTR& operator=(const dotk::DOTk_MexNewtonTypeTR & rhs_);
};

}

#endif /* DOTK_MEXNEWTONTYPETR_HPP_ */
