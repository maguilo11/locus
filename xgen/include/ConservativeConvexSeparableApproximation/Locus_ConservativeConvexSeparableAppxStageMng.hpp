/*
 * Locus_ConservativeConvexSeparableAppxStageMng.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_CONSERVATIVECONVEXSEPARABLEAPPXSTAGEMNG_HPP_
#define LOCUS_CONSERVATIVECONVEXSEPARABLEAPPXSTAGEMNG_HPP_

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class MultiVector;
template<typename ScalarType, typename OrdinalType>
class ConservativeConvexSeparableAppxDataMng;

template<typename ScalarType, typename OrdinalType = size_t>
class ConservativeConvexSeparableAppxStageMng
{
public:
    virtual ~ConservativeConvexSeparableAppxStageMng()
    {
    }

    virtual void update(const locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
};

} // namespace locus

#endif /* LOCUS_CONSERVATIVECONVEXSEPARABLEAPPXSTAGEMNG_HPP_ */
