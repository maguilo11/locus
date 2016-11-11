/*
 * DOTk_BoundConstraintFactory.hpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_BOUNDCONSTRAINTFACTORY_HPP_
#define DOTK_BOUNDCONSTRAINTFACTORY_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_BoundConstraint;

class DOTk_BoundConstraintFactory
{
public:
    DOTk_BoundConstraintFactory();
    ~DOTk_BoundConstraintFactory();

    void setFactoryType(dotk::types::constraint_method_t type_);
    dotk::types::constraint_method_t getFactoryType() const;

    void buildFeasibleDirection(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                std::tr1::shared_ptr<dotk::DOTk_BoundConstraint> & bound_);
    void buildProjectionAlongFeasibleDirection(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                               std::tr1::shared_ptr<dotk::DOTk_BoundConstraint> & bound_);

    void build(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
               std::tr1::shared_ptr<dotk::DOTk_BoundConstraint> & bound_) const;

private:
    dotk::types::constraint_method_t m_Type;

private:
    DOTk_BoundConstraintFactory(const dotk::DOTk_BoundConstraintFactory &);
    dotk::DOTk_BoundConstraintFactory operator=(const dotk::DOTk_BoundConstraintFactory &);
};

}

#endif /* DOTK_BOUNDCONSTRAINTFACTORY_HPP_ */