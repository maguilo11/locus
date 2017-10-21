/*
 * Locus_GradientOperatorList.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_GRADIENTOPERATORLIST_HPP_
#define LOCUS_GRADIENTOPERATORLIST_HPP_

#include <vector>
#include <memory>
#include <cassert>

#include "Locus_GradientOperator.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class GradientOperatorList
{
public:
    GradientOperatorList() :
            mList()
    {
    }
    ~GradientOperatorList()
    {
    }

    OrdinalType size() const
    {
        return (mList.size());
    }
    void add(const locus::GradientOperator<ScalarType, OrdinalType> & aInput)
    {
        mList.push_back(aInput.create());
    }
    void add(const std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> & aInput)
    {
        mList.push_back(aInput);
    }
    locus::GradientOperator<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex)
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    const locus::GradientOperator<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    std::shared_ptr<locus::GradientOperatorList<ScalarType, OrdinalType>> create() const
    {
        assert(this->size() > static_cast<OrdinalType>(0));

        std::shared_ptr<locus::GradientOperatorList<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::GradientOperatorList<ScalarType, OrdinalType>>();
        const OrdinalType tNumGradientOperators = this->size();
        for(OrdinalType tIndex = 0; tIndex < tNumGradientOperators; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);

            const std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> & tGradientOperator = mList[tIndex];
            tOutput->add(tGradientOperator);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>> & ptr(const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return(mList[aIndex]);
    }

private:
    std::vector<std::shared_ptr<locus::GradientOperator<ScalarType, OrdinalType>>> mList;

private:
    GradientOperatorList(const locus::GradientOperatorList<ScalarType, OrdinalType>&);
    locus::GradientOperatorList<ScalarType, OrdinalType> & operator=(const locus::GradientOperatorList<ScalarType, OrdinalType>&);
};

} //namespace locus

#endif /* LOCUS_GRADIENTOPERATORLIST_HPP_ */
