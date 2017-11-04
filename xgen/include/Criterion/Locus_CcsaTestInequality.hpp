/*
 * Locus_CcsaTestInequality.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CCSATESTINEQUALITY_HPP_
#define LOCUS_CCSATESTINEQUALITY_HPP_

#include <cmath>
#include <vector>
#include <memory>
#include <cassert>

#include "Locus_Vector.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class CcsaTestInequality : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    CcsaTestInequality() :
            mConstant(1)
    {
    }
    virtual ~CcsaTestInequality()
    {
    }

    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                     const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));

        const OrdinalType tVectorIndex = 0;
        const locus::Vector<ScalarType, OrdinalType> & tMyControl = aControl[tVectorIndex];

        ScalarType tTermOne = static_cast<ScalarType>(61) / std::pow(tMyControl[0], static_cast<ScalarType>(3));
        ScalarType tTermTwo = static_cast<ScalarType>(37) / std::pow(tMyControl[1], static_cast<ScalarType>(3));
        ScalarType tTermThree = static_cast<ScalarType>(19) / std::pow(tMyControl[2], static_cast<ScalarType>(3));
        ScalarType tTermFour = static_cast<ScalarType>(7) / std::pow(tMyControl[3], static_cast<ScalarType>(3));
        ScalarType tTermFive = static_cast<ScalarType>(1) / std::pow(tMyControl[4], static_cast<ScalarType>(3));
        ScalarType tResidual = tTermOne + tTermTwo + tTermThree + tTermFour + tTermFive;
        tResidual = tResidual - mConstant;

        return (tResidual);
    }
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                  const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));

        const OrdinalType tVectorIndex = 0;
        locus::Vector<ScalarType, OrdinalType> & tMyGradient = aOutput[tVectorIndex];
        const locus::Vector<ScalarType, OrdinalType> & tMyControl = aControl[tVectorIndex];

        ScalarType tScaleFactor = -3;
        tMyGradient[0] = tScaleFactor
                * (static_cast<ScalarType>(61) / std::pow(tMyControl[0], static_cast<ScalarType>(4)));
        tMyGradient[1] = tScaleFactor
                * (static_cast<ScalarType>(37) / std::pow(tMyControl[1], static_cast<ScalarType>(4)));
        tMyGradient[2] = tScaleFactor
                * (static_cast<ScalarType>(19) / std::pow(tMyControl[2], static_cast<ScalarType>(4)));
        tMyGradient[3] = tScaleFactor
                * (static_cast<ScalarType>(7) / std::pow(tMyControl[3], static_cast<ScalarType>(4)));
        tMyGradient[4] = tScaleFactor
                * (static_cast<ScalarType>(1) / std::pow(tMyControl[4], static_cast<ScalarType>(4)));
    }

    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::CcsaTestInequality<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    ScalarType mConstant;

private:
    CcsaTestInequality(const locus::CcsaTestInequality<ScalarType, OrdinalType> & aRhs);
    locus::CcsaTestInequality<ScalarType, OrdinalType> & operator=(const locus::CcsaTestInequality<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_CCSATESTINEQUALITY_HPP_ */
