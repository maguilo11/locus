/*
 * Locus_Radius.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_RADIUS_HPP_
#define LOCUS_RADIUS_HPP_

#include <cmath>
#include <memory>
#include <cassert>

#include "Locus_Vector.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class Radius : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    Radius() :
            mLimit(1)
    {
    }
    virtual ~Radius()
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
        ScalarType tOutput = std::pow(aControl(tVectorIndex, 0), static_cast<ScalarType>(2.)) +
                std::pow(aControl(tVectorIndex, 1), static_cast<ScalarType>(2.));
        tOutput = tOutput - mLimit;
        return (tOutput);
    }
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aOutput.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));
        assert(aControl.getNumVectors() == aOutput.getNumVectors());

        const OrdinalType tVectorIndex = 0;
        aOutput(tVectorIndex, 0) = static_cast<ScalarType>(2.) * aControl(tVectorIndex, 0);
        aOutput(tVectorIndex, 1) = static_cast<ScalarType>(2.) * aControl(tVectorIndex, 1);

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
        aOutput(tVectorIndex, 1) = static_cast<ScalarType>(2.) * aVector(tVectorIndex, 1);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::Radius<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    ScalarType mLimit;

private:
    Radius(const locus::Radius<ScalarType, OrdinalType> & aRhs);
    locus::Radius<ScalarType, OrdinalType> & operator=(const locus::Radius<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_RADIUS_HPP_ */
