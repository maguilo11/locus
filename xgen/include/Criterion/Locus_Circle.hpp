/*
 * Locus_Circle.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CIRCLE_HPP_
#define LOCUS_CIRCLE_HPP_

#include <cmath>
#include <memory>
#include <cassert>

#include "Locus_Vector.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class Circle : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    Circle()
    {
    }
    virtual ~Circle()
    {
    }

    void cacheData()
    {
        return;
    }
    /// \left(\mathbf{z}(0) - 1.\right)^2 + 2\left(\mathbf{z}(1) - 2\right)^2
    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));

        const OrdinalType tVectorIndex = 0;
        ScalarType tAlpha = aControl(tVectorIndex, 0) - static_cast<ScalarType>(1.);
        ScalarType tBeta = aControl(tVectorIndex, 1) - static_cast<ScalarType>(2);
        tBeta = static_cast<ScalarType>(2.) * std::pow(tBeta, static_cast<ScalarType>(2));
        ScalarType tOutput = std::pow(tAlpha, static_cast<ScalarType>(2)) + tBeta;
        return (tOutput);
    }
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aControl.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) =
                static_cast<ScalarType>(2.) * (aControl(tVectorIndex, 0) - static_cast<ScalarType>(1.));
        aOutput(tVectorIndex, 1) =
                static_cast<ScalarType>(4.) * (aControl(tVectorIndex, 1) - static_cast<ScalarType>(2.));

    }
    void hessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                 const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                 locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aVector.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aVector.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) = static_cast<ScalarType>(2.) * aVector(tVectorIndex, 0);
        aOutput(tVectorIndex, 1) = static_cast<ScalarType>(4.) * aVector(tVectorIndex, 1);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::Circle<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    Circle(const locus::Circle<ScalarType, OrdinalType> & aRhs);
    locus::Circle<ScalarType, OrdinalType> & operator=(const locus::Circle<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_CIRCLE_HPP_ */
