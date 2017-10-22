/*
 * Locus_FletcherReeves.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_FLETCHERREEVES_HPP_
#define LOCUS_FLETCHERREEVES_HPP_

#include <cmath>
#include <limits>
#include <memory>

#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_NonlinearConjugateGradientStep.hpp"
#include "Locus_NonlinearConjugateGradientDataMng.hpp"
#include "Locus_NonlinearConjugateGradientStageMngBase.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class FletcherReeves : public locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>
{
public:
    FletcherReeves(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mScaledDescentDirection(aDataFactory.control().create())
    {
    }
    virtual ~FletcherReeves()
    {
    }

    void computeScaledDescentDirection(locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & aDataMng,
                                       locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & aStageMng)
    {
        locus::update(static_cast<ScalarType>(1),
                      aDataMng.getTrialStep(),
                      static_cast<ScalarType>(0),
                      mScaledDescentDirection.operator*());

        ScalarType tBeta = locus::dot(aDataMng.getCurrentGradient(), aDataMng.getCurrentGradient())
                / locus::dot(aDataMng.getPreviousGradient(), aDataMng.getPreviousGradient());
        tBeta = std::max(tBeta, std::numeric_limits<ScalarType>::min());

        locus::update(static_cast<ScalarType>(-1),
                      aDataMng.getCurrentGradient(),
                      tBeta,
                      mScaledDescentDirection.operator*());

        aDataMng.setTrialStep(mScaledDescentDirection.operator*());
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mScaledDescentDirection;

private:
    FletcherReeves(const locus::FletcherReeves<ScalarType, OrdinalType> & aRhs);
    locus::FletcherReeves<ScalarType, OrdinalType> & operator=(const locus::FletcherReeves<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_FLETCHERREEVES_HPP_ */
