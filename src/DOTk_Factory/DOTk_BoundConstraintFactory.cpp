/*
 * DOTk_BoundConstraintFactory.cpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <iostream>

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

void DOTk_BoundConstraintFactory::setFactoryType(dotk::types::constraint_method_t aType)
{
    m_Type = aType;
}

dotk::types::constraint_method_t DOTk_BoundConstraintFactory::getFactoryType() const
{
    return (m_Type);
}

void DOTk_BoundConstraintFactory::buildFeasibleDirection
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 std::shared_ptr<dotk::DOTk_BoundConstraint> & aBound)
{
    this->setFactoryType(dotk::types::FEASIBLE_DIR);
    aBound = std::make_shared<dotk::DOTk_FeasibleDirection>(aPrimal);
}

void DOTk_BoundConstraintFactory::buildProjectionAlongFeasibleDirection
(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
 std::shared_ptr<dotk::DOTk_BoundConstraint> & aBound)
{
    this->setFactoryType(dotk::types::PROJECTION_ALONG_FEASIBLE_DIR);
    aBound = std::make_shared<dotk::DOTk_ProjectionAlongFeasibleDir>(aPrimal);
    aBound->setStepType(dotk::types::CONSTANT_STEP);
}

void DOTk_BoundConstraintFactory::build(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                        std::shared_ptr<dotk::DOTk_BoundConstraint> & aBound) const
{
    switch(this->getFactoryType())
    {
        case dotk::types::FEASIBLE_DIR:
        {
            aBound = std::make_shared<dotk::DOTk_FeasibleDirection>(aPrimal);
            break;
        }
        case dotk::types::PROJECTION_ALONG_FEASIBLE_DIR:
        {
            aBound = std::make_shared<dotk::DOTk_ProjectionAlongFeasibleDir>(aPrimal);
            aBound->setStepType(dotk::types::CONSTANT_STEP);
            break;
        }
        case dotk::types::CONSTRAINT_METHOD_DISABLED:
        default:
        {
            std::cout << "\nDOTk WARNING: Invalid bound constraint type, Default bound constraint method set to "
                    << "ARMIJO RULE ALONG THE PROJECTION ARC.\n" << std::flush;
            aBound = std::make_shared<dotk::DOTk_ProjectionAlongFeasibleDir>(aPrimal);
            break;
        }
    }
}

}
