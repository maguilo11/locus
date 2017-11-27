/*
 * Locus_OptimalityCriteriaTestInequalityTwo.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_OPTIMALITYCRITERIATESTINEQUALITYTWO_HPP_
#define LOCUS_OPTIMALITYCRITERIATESTINEQUALITYTWO_HPP_

#include <cmath>
#include <memory>
#include <cassert>

#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class OptimalityCriteriaTestInequalityTwo : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    explicit OptimalityCriteriaTestInequalityTwo()
    {
    }
    virtual ~OptimalityCriteriaTestInequalityTwo()
    {
    }

    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        const OrdinalType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<OrdinalType>(2));

        ScalarType tDenominator = aControl(tVectorIndex, 1) + (static_cast<ScalarType>(0.25) * aControl(tVectorIndex, 0));
        ScalarType tOutput = static_cast<ScalarType>(1) - (static_cast<ScalarType>(1.5) / tDenominator);

        return (tOutput);
    }

    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aGradient)
    {
        const OrdinalType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<OrdinalType>(2));

        ScalarType tPower = 2;
        ScalarType tDenominator = aControl(tVectorIndex, 1) + (static_cast<ScalarType>(0.25) * aControl(tVectorIndex, 0));
        tDenominator = std::pow(tDenominator, tPower);
        ScalarType tFirstElement = static_cast<ScalarType>(0.375) / tDenominator;
        aGradient(tVectorIndex, 0) = tFirstElement;
        ScalarType tSecondElement = static_cast<ScalarType>(1.5) / tDenominator;
        aGradient(tVectorIndex, 1) = tSecondElement;
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::OptimalityCriteriaTestInequalityTwo<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    OptimalityCriteriaTestInequalityTwo(const locus::OptimalityCriteriaTestInequalityTwo<ScalarType, OrdinalType>&);
    locus::OptimalityCriteriaTestInequalityTwo<ScalarType, OrdinalType> & operator=(const locus::OptimalityCriteriaTestInequalityTwo<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_OPTIMALITYCRITERIATESTINEQUALITYTWO_HPP_ */
