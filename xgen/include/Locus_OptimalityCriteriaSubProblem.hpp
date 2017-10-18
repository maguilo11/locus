/*
 * Locus_OptimalityCriteriaSubProblem.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_OPTIMALITYCRITERIASUBPROBLEM_HPP_
#define LOCUS_OPTIMALITYCRITERIASUBPROBLEM_HPP_

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class OptimalityCriteriaDataMng;
template<typename ScalarType, typename OrdinalType>
class OptimalityCriteriaStageMngBase;

template<typename ScalarType, typename OrdinalType = size_t>
class OptimalityCriteriaSubProblem
{
public:
    virtual ~OptimalityCriteriaSubProblem(){}

    virtual void solve(locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType> & aDataMng,
                       locus::OptimalityCriteriaStageMngBase<ScalarType, OrdinalType> & aStageMng) = 0;
};

}

#endif /* LOCUS_OPTIMALITYCRITERIASUBPROBLEM_HPP_ */
