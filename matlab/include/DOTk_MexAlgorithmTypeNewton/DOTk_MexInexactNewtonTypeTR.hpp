/*
 * DOTk_MexInexactNewtonTypeTR.hpp
 *
 *  Created on: Apr 25, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXINEXACTNEWTONTYPETR_HPP_
#define DOTK_MEXINEXACTNEWTONTYPETR_HPP_

#include <memory>

#include "DOTk_MexAlgorithmTypeNewton.hpp"

namespace dotk
{

class DOTk_TrustRegionInexactNewton;
class DOTk_TrustRegionAlgorithmsDataMng;

class DOTk_MexInexactNewtonTypeTR : public dotk::DOTk_MexAlgorithmTypeNewton
{
public:
    explicit DOTk_MexInexactNewtonTypeTR(const mxArray* options_[]);
    ~DOTk_MexInexactNewtonTypeTR();

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
    DOTk_MexInexactNewtonTypeTR(const dotk::DOTk_MexInexactNewtonTypeTR & rhs_);
    dotk::DOTk_MexInexactNewtonTypeTR& operator=(const dotk::DOTk_MexInexactNewtonTypeTR & rhs_);
};

}

#endif /* DOTK_MEXINEXACTNEWTONTYPETR_HPP_ */
