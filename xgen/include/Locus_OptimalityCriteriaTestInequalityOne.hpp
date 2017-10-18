/*
 * Locus_OptimalityCriteriaTestInequalityOne.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_OPTIMALITYCRITERIATESTINEQUALITYONE_HPP_
#define LOCUS_OPTIMALITYCRITERIATESTINEQUALITYONE_HPP_


#include <cmath>
#include <memory>
#include <cassert>

#include <Locus_Criterion.hpp>
#include <Locus_MultiVector.hpp>

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class OptimalityCriteriaTestInequalityOne : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    explicit OptimalityCriteriaTestInequalityOne() :
            mBound(1.)
    {
    }
    virtual ~OptimalityCriteriaTestInequalityOne()
    {
    }

    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                     const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        const OrdinalType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<OrdinalType>(5));

        const ScalarType tPower = 3.;
        ScalarType tFirstTerm = static_cast<ScalarType>(61.) / std::pow(aControl(tVectorIndex,0), tPower);
        ScalarType tSecondTerm = static_cast<ScalarType>(37.) / std::pow(aControl(tVectorIndex,1), tPower);
        ScalarType tThirdTerm = static_cast<ScalarType>(19.) / std::pow(aControl(tVectorIndex,2), tPower);
        ScalarType tFourthTerm = static_cast<ScalarType>(7.) / std::pow(aControl(tVectorIndex,3), tPower);
        ScalarType tFifthTerm = static_cast<ScalarType>(1.) / std::pow(aControl(tVectorIndex,4), tPower);

        ScalarType tValue = tFirstTerm + tSecondTerm + tThirdTerm + tFourthTerm + tFifthTerm;
        ScalarType tOutput = tValue - mBound;

        return (tOutput);
    }

    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                  const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aGradient)
    {
        const OrdinalType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<OrdinalType>(5));

        const ScalarType tPower = 4;
        const ScalarType tScaleFactor = -3.;
        aGradient(tVectorIndex,0) = tScaleFactor * (static_cast<ScalarType>(61.) / std::pow(aControl(tVectorIndex,0), tPower));
        aGradient(tVectorIndex,1) = tScaleFactor * (static_cast<ScalarType>(37.) / std::pow(aControl(tVectorIndex,1), tPower));
        aGradient(tVectorIndex,2) = tScaleFactor * (static_cast<ScalarType>(19.) / std::pow(aControl(tVectorIndex,2), tPower));
        aGradient(tVectorIndex,3) = tScaleFactor * (static_cast<ScalarType>(7.) / std::pow(aControl(tVectorIndex,3), tPower));
        aGradient(tVectorIndex,4) = tScaleFactor * (static_cast<ScalarType>(1.) / std::pow(aControl(tVectorIndex,4), tPower));
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::OptimalityCriteriaTestInequalityOne<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    ScalarType mBound;

private:
    OptimalityCriteriaTestInequalityOne(const locus::OptimalityCriteriaTestInequalityOne<ScalarType, OrdinalType>&);
    locus::OptimalityCriteriaTestInequalityOne<ScalarType, OrdinalType> & operator=(const locus::OptimalityCriteriaTestInequalityOne<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_OPTIMALITYCRITERIATESTINEQUALITYONE_HPP_ */
