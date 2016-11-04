/*
 * DOTk_BoundConstraintFactory.cpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_FeasibleDirection.hpp"
#include "DOTk_BoundConstraintFactory.hpp"
#include "DOTk_ProjectionAlongFeasibleDir.hpp"

namespace dotk
{

DOTk_BoundConstraintFactory::DOTk_BoundConstraintFactory() :
        m_Type(dotk::types::CONSTRAINT_METHOD_DISABLED)
{
}

DOTk_BoundConstraintFactory::~DOTk_BoundConstraintFactory()
{
}

void DOTk_BoundConstraintFactory::setFactoryType(dotk::types::constraint_method_t type_)
{
    m_Type = type_;
}

dotk::types::constraint_method_t DOTk_BoundConstraintFactory::getFactoryType() const
{
    return (m_Type);
}

void DOTk_BoundConstraintFactory::buildFeasibleDirection
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::tr1::shared_ptr<dotk::DOTk_BoundConstraint> & bound_)
{
    this->setFactoryType(dotk::types::FEASIBLE_DIR);
    bound_.reset(new dotk::DOTk_FeasibleDirection(primal_));
}

void DOTk_BoundConstraintFactory::buildProjectionAlongFeasibleDirection
(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
 std::tr1::shared_ptr<dotk::DOTk_BoundConstraint> & bound_)
{
    this->setFactoryType(dotk::types::PROJECTION_ALONG_FEASIBLE_DIR);
    bound_.reset(new dotk::DOTk_ProjectionAlongFeasibleDir(primal_));
    bound_->setStepType(dotk::types::CONSTANT_STEP);
}

void DOTk_BoundConstraintFactory::build(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                        std::tr1::shared_ptr<dotk::DOTk_BoundConstraint> & bound_) const
{
    switch(this->getFactoryType())
    {
        case dotk::types::FEASIBLE_DIR:
        {
            bound_.reset(new dotk::DOTk_FeasibleDirection(primal_));
            break;
        }
        case dotk::types::PROJECTION_ALONG_FEASIBLE_DIR:
        {
            bound_.reset(new dotk::DOTk_ProjectionAlongFeasibleDir(primal_));
            bound_->setStepType(dotk::types::CONSTANT_STEP);
            break;
        }
        case dotk::types::CONSTRAINT_METHOD_DISABLED:
        default:
        {
            std::cout << "\nDOTk WARNING: Invalid bound constraint type, Default bound constraint method set to "
                    << "ARMIJO RULE ALONG THE PROJECTION ARC.\n" << std::flush;
            bound_.reset(new dotk::DOTk_ProjectionAlongFeasibleDir(primal_));
            break;
        }
    }
}

}
