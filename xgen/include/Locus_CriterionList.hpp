/*
 * Locus_CriterionList.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CRITERIONLIST_HPP_
#define LOCUS_CRITERIONLIST_HPP_

#include <vector>
#include <memory>
#include <cassert>

#include "Locus_Criterion.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class CriterionList
{
public:
    CriterionList() :
            mList()
    {
    }
    ~CriterionList()
    {
    }

    OrdinalType size() const
    {
        return (mList.size());
    }
    void add(const locus::Criterion<ScalarType, OrdinalType> & aCriterion)
    {
        mList.push_back(aCriterion.create());
    }
    void add(const std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> & aCriterion)
    {
        mList.push_back(aCriterion);
    }
    locus::Criterion<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex)
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    const locus::Criterion<ScalarType, OrdinalType> & operator [](const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return (mList[aIndex].operator*());
    }
    std::shared_ptr<locus::CriterionList<ScalarType, OrdinalType>> create() const
    {
        assert(this->size() > static_cast<OrdinalType>(0));
        std::shared_ptr<locus::CriterionList<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::CriterionList<ScalarType, OrdinalType>>();
        const OrdinalType tNumCriterion = this->size();
        for(OrdinalType tIndex = 0; tIndex < tNumCriterion; tIndex++)
        {
            assert(mList[tIndex].get() != nullptr);
            const std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> & tCriterion = mList[tIndex];
            tOutput->add(tCriterion);
        }
        return (tOutput);
    }
    const std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> & ptr(const OrdinalType & aIndex) const
    {
        assert(aIndex < mList.size());
        assert(mList[aIndex].get() != nullptr);
        return(mList[aIndex]);
    }

private:
    std::vector<std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>>> mList;

private:
    CriterionList(const locus::CriterionList<ScalarType, OrdinalType>&);
    locus::CriterionList<ScalarType, OrdinalType> & operator=(const locus::CriterionList<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_CRITERIONLIST_HPP_ */
