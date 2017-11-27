/*
 * Locus_OptimalityCriteriaTestObjectiveTwo.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_OPTIMALITYCRITERIATESTOBJECTIVETWO_HPP_
#define LOCUS_OPTIMALITYCRITERIATESTOBJECTIVETWO_HPP_

#include <memory>
#include <cassert>

#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class OptimalityCriteriaTestObjectiveTwo : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    OptimalityCriteriaTestObjectiveTwo()
    {
    }
    virtual ~OptimalityCriteriaTestObjectiveTwo()
    {
    }

    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        const OrdinalType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<OrdinalType>(2));
        ScalarType tOutput = aControl(tVectorIndex, 0) + (static_cast<ScalarType>(2) * aControl(tVectorIndex, 1));
        return (tOutput);
    }

    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aGradient)
    {
        const OrdinalType tVectorIndex = 0;
        assert(aControl[tVectorIndex].size() == static_cast<OrdinalType>(2));
        aGradient(tVectorIndex, 0) = static_cast<ScalarType>(1);
        aGradient(tVectorIndex, 1) = static_cast<ScalarType>(2);
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::OptimalityCriteriaTestObjectiveTwo<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    OptimalityCriteriaTestObjectiveTwo(const locus::OptimalityCriteriaTestObjectiveTwo<ScalarType, OrdinalType>&);
    locus::OptimalityCriteriaTestObjectiveTwo<ScalarType, OrdinalType> & operator=(const locus::OptimalityCriteriaTestObjectiveTwo<
            ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_OPTIMALITYCRITERIATESTOBJECTIVETWO_HPP_ */
