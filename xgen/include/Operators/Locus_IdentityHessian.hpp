/*
 * Locus_IdentityHessian.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_IDENTITYHESSIAN_HPP_
#define LOCUS_IDENTITYHESSIAN_HPP_

#include <memory>

#include "Locus_StateData.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_LinearOperator.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class IdentityHessian : public locus::LinearOperator<ScalarType, OrdinalType>
{
public:
    IdentityHessian()
    {
    }
    virtual ~IdentityHessian()
    {
    }

    void update(const locus::StateData<ScalarType, OrdinalType> & aStateData)
    {
        return;
    }
    void apply(const locus::MultiVector<ScalarType, OrdinalType> & aState,
               const locus::MultiVector<ScalarType, OrdinalType> & aControl,
               const locus::MultiVector<ScalarType, OrdinalType> & aVector,
               locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
        locus::update(static_cast<ScalarType>(1), aVector, static_cast<ScalarType>(0), aOutput);
    }
    std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::LinearOperator<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::IdentityHessian<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    IdentityHessian(const locus::IdentityHessian<ScalarType, OrdinalType> & aRhs);
    locus::IdentityHessian<ScalarType, OrdinalType> & operator=(const locus::IdentityHessian<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_IDENTITYHESSIAN_HPP_ */
