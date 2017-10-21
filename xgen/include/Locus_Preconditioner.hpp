/*
 * Locus_Preconditioner.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_PRECONDITIONER_HPP_
#define LOCUS_PRECONDITIONER_HPP_

#include <memory>

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class StateData;
template<typename ScalarType, typename OrdinalType>
class MultiVector;

template<typename ScalarType, typename OrdinalType = size_t>
class Preconditioner
{
public:
    virtual ~Preconditioner()
    {
    }

    virtual void update(const locus::StateData<ScalarType, OrdinalType> & aStateData) = 0;
    virtual void applyPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                     const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                     locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyInvPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                        const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                        locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual std::shared_ptr<locus::Preconditioner<ScalarType, OrdinalType>> create() const = 0;
};

}

#endif /* LOCUS_PRECONDITIONER_HPP_ */
