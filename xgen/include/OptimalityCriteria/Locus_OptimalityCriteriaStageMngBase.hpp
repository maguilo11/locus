/*
 * Locus_OptimalityCriteriaStageMngBase.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_OPTIMALITYCRITERIASTAGEMNGBASE_HPP_
#define LOCUS_OPTIMALITYCRITERIASTAGEMNGBASE_HPP_

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class MultiVector;
template<typename ScalarType, typename OrdinalType>
class OptimalityCriteriaDataMng;

template<typename ScalarType, typename OrdinalType = size_t>
class OptimalityCriteriaStageMngBase
{
public:
    virtual ~OptimalityCriteriaStageMngBase()
    {
    }

    virtual void update(locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
};

}

#endif /* LOCUS_OPTIMALITYCRITERIASTAGEMNGBASE_HPP_ */
