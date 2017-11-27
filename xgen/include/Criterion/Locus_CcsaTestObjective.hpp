/*
 * Locus_CcsaTestObjective.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CCSATESTOBJECTIVE_HPP_
#define LOCUS_CCSATESTOBJECTIVE_HPP_

#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>

#include "Locus_Vector.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_ReductionOperations.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class CcsaTestObjective : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    explicit CcsaTestObjective(const locus::ReductionOperations<ScalarType, OrdinalType> & aReduction) :
            mConstant(0.0624),
            mReduction(aReduction.create())
    {
    }
    virtual ~CcsaTestObjective()
    {
    }

    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        assert(aControl.getNumVectors() > static_cast<OrdinalType>(0));

        OrdinalType tNumVectors = aControl.getNumVectors();
        std::vector<ScalarType> tStorage(tNumVectors);
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tMyControl = aControl[tVectorIndex];
            tStorage[tVectorIndex] = mReduction->sum(tMyControl);
        }
        const ScalarType tInitialValue = 0;
        ScalarType tSum = std::accumulate(tStorage.begin(), tStorage.end(), tInitialValue);
        ScalarType tOutput = mConstant * tSum;

        return (tOutput);
    }
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        locus::fill(mConstant, aOutput);
    }

    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::CcsaTestObjective<ScalarType, OrdinalType>>(mReduction.operator*());
        return (tOutput);
    }

private:
    ScalarType mConstant;
    std::shared_ptr<locus::ReductionOperations<ScalarType, OrdinalType>> mReduction;

private:
    CcsaTestObjective(const locus::CcsaTestObjective<ScalarType, OrdinalType> & aRhs);
    locus::CcsaTestObjective<ScalarType, OrdinalType> & operator=(const locus::CcsaTestObjective<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_CCSATESTOBJECTIVE_HPP_ */
