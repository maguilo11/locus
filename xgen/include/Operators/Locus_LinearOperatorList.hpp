/*
 * Locus_LinearOperatorList.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_LINEAROPERATORLIST_HPP_
#define LOCUS_LINEAROPERATORLIST_HPP_

#include <vector>
#include <memory>
#include <cassert>

#include "Locus_LinearOperator.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class LinearOperatorList
{
public:
    LinearOperatorList() :
            mList()
    {
    }
    ~LinearOperatorList()
    {
    }

    OrdinalType size() const
    {
        return (mList.size());
    }
    void add(const locus::LinearOperator<ScalarType, OrdinalType> & aInput)
    {
        mList.push_back(aInput.create());
    }
    void add(const std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> & aInput)
    {
        mList.push_back(aInput);
    }
    locus::LinearOperator<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex)
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    const locus::LinearOperator<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    std::shared_ptr<locus::LinearOperatorList<ScalarType, OrdinalType>> create() const
    {
        assert(this->size() > static_cast<OrdinalType>(0));

        std::shared_ptr<locus::LinearOperatorList<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::LinearOperatorList<ScalarType, OrdinalType>>();
        const OrdinalType tNumLinearOperators = this->size();
        for(OrdinalType tIndex = 0; tIndex < tNumLinearOperators; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);

            const std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> & tLinearOperator = mList[tIndex];
            tOutput->add(tLinearOperator);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> & ptr(const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return(mList[aIndex]);
    }

private:
    std::vector<std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>>> mList;

private:
    LinearOperatorList(const locus::LinearOperatorList<ScalarType, OrdinalType>&);
    locus::LinearOperatorList<ScalarType, OrdinalType> & operator=(const locus::LinearOperatorList<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_LINEAROPERATORLIST_HPP_ */
