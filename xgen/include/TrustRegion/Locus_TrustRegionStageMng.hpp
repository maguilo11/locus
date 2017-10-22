/*
 * Locus_TrustRegionStageMng.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_TRUSTREGIONSTAGEMNG_HPP_
#define LOCUS_TRUSTREGIONSTAGEMNG_HPP_

#include <limits>

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class MultiVector;
template<typename ScalarType, typename OrdinalType>
class TrustRegionAlgorithmDataMng;

template<typename ScalarType, typename OrdinalType = size_t>
class TrustRegionStageMng
{
public:
    virtual ~TrustRegionStageMng()
    {
    }

    virtual void update(const locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng) = 0;
    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                          ScalarType aTolerance = std::numeric_limits<ScalarType>::max()) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                      const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                      locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                             const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                             locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToInvPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                                const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                                locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
};

} // namespace locus

#endif /* LOCUS_TRUSTREGIONSTAGEMNG_HPP_ */
