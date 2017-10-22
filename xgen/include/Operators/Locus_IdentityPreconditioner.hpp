/*
 * Locus_IdentityPreconditioner.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_IDENTITYPRECONDITIONER_HPP_
#define LOCUS_IDENTITYPRECONDITIONER_HPP_

#include "Locus_StateData.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_Preconditioner.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class IdentityPreconditioner : public Preconditioner<ScalarType, OrdinalType>
{
public:
    IdentityPreconditioner()
    {
    }
    virtual ~IdentityPreconditioner()
    {
    }
    void update(const locus::StateData<ScalarType, OrdinalType> & aStateData)
    {
        return;
    }
    void applyPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                             const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                             locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aVector.getNumVectors() == aOutput.getNumVectors());
        locus::update(1., aVector, 0., aOutput);
    }
    void applyInvPreconditioner(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        assert(aVector.getNumVectors() == aOutput.getNumVectors());
        locus::update(1., aVector, 0., aOutput);
    }
    std::shared_ptr<locus::Preconditioner<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Preconditioner<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::IdentityPreconditioner<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    IdentityPreconditioner(const locus::IdentityPreconditioner<ScalarType, OrdinalType> & aRhs);
    locus::IdentityPreconditioner<ScalarType, OrdinalType> & operator=(const locus::IdentityPreconditioner<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_IDENTITYPRECONDITIONER_HPP_ */
