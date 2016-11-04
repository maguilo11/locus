/*
 * DOTk_OrthogonalFactorization.cpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_OrthogonalFactorization.hpp"

namespace dotk
{

DOTk_OrthogonalFactorization::DOTk_OrthogonalFactorization(dotk::types::qr_t type_) :
        m_OrthogonalFactorizationType(type_)
{
}
DOTk_OrthogonalFactorization::~DOTk_OrthogonalFactorization()
{
}

dotk::types::qr_t DOTk_OrthogonalFactorization::type() const
{
    return (m_OrthogonalFactorizationType);
}

}
