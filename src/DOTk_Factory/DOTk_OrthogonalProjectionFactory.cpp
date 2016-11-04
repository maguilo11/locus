/*
 * DOTk_OrthogonalProjectionFactory.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "DOTk_Primal.hpp"
#include "DOTk_GramSchmidt.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_ArnoldiProjection.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_OrthogonalProjectionFactory.hpp"

namespace dotk
{

DOTk_OrthogonalProjectionFactory::DOTk_OrthogonalProjectionFactory(size_t krylov_subspace_dim_,
                                                                   dotk::types::projection_t type_) :
    m_WarningMsg(),
    m_KrylovSubspaceDimension(krylov_subspace_dim_),
    m_FactoryType(type_)
{
}

DOTk_OrthogonalProjectionFactory::~DOTk_OrthogonalProjectionFactory()
{
}

std::string DOTk_OrthogonalProjectionFactory::getWarningMsg() const
{
    return (m_WarningMsg);
}

size_t DOTk_OrthogonalProjectionFactory::getKrylovSubspaceDim() const
{
    return (m_KrylovSubspaceDimension);
}

dotk::types::projection_t DOTk_OrthogonalProjectionFactory::getFactoryType() const
{
    return (m_FactoryType);
}

void DOTk_OrthogonalProjectionFactory::buildGramSchmidt
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & projection_)
{
    this->setFactoryType(dotk::types::GRAM_SCHMIDT);
    size_t krylov_subspace_dim = this->getKrylovSubspaceDim();
    projection_.reset(new dotk::DOTk_GramSchmidt(primal_, krylov_subspace_dim));
}

void DOTk_OrthogonalProjectionFactory::buildArnoldiProjection
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & projection_)
{
    this->setFactoryType(dotk::types::ARNOLDI);
    size_t krylov_subspace_dim = this->getKrylovSubspaceDim();
    projection_.reset(new dotk::DOTk_ArnoldiProjection(primal_, krylov_subspace_dim));
}

void DOTk_OrthogonalProjectionFactory::build
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & projection_)
{
    switch (this->getFactoryType())
    {
        case dotk::types::ARNOLDI:
        {
            this->buildArnoldiProjection(primal_, projection_);
            break;
        }
        case dotk::types::GRAM_SCHMIDT:
        {
            this->buildGramSchmidt(primal_, projection_);
            break;
        }
        case dotk::types::MODIFIED_GRAM_SCHMIDT:
        case dotk::types::HOUSEHOLDER:
        case dotk::types::ARNOLDI_MODIFIED_GRAM_SCHMIDT:
        case dotk::types::ARNOLDI_HOUSEHOLDER:
        case dotk::types::INCOMPLETE_ORTHOGONALIZATION:
        case dotk::types::DIRECT_IMCOMPLETE_ORTHOGONALIZATION:
        default:
        {
            std::ostringstream msg;
            msg << "\nDOTk WARNING: Invalid orthogonal projection method."
                    << " Default method will be set to Gram-Schmidt.\n" << std::flush;
            dotk::ioUtils::printMessage(msg);
            this->setWarningMsg(msg.str());
            this->buildGramSchmidt(primal_, projection_);
            break;
        }
    }
}

void DOTk_OrthogonalProjectionFactory::setWarningMsg(const std::string & msg_)
{
    m_WarningMsg.append(msg_);
}

void DOTk_OrthogonalProjectionFactory::setFactoryType(dotk::types::projection_t type_)
{
    m_FactoryType = type_;
}

}
