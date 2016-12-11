/*
 * DOTk_RightPreconditionerFactory.cpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Types.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_RightPreconditioner.hpp"
#include "DOTk_RightPreconditionerFactory.hpp"

namespace dotk
{

DOTk_RightPreconditionerFactory::DOTk_RightPreconditionerFactory(dotk::types::right_prec_t type_) :
        mWarningMsg(),
        mFactoryType(type_)
{
}

DOTk_RightPreconditionerFactory::~DOTk_RightPreconditionerFactory()
{
}

void DOTk_RightPreconditionerFactory::setWarningMsg(const std::string & msg_)
{
    mWarningMsg.append(msg_);
}

std::string DOTk_RightPreconditionerFactory::getWarningMsg() const
{
    return (mWarningMsg);
}

void DOTk_RightPreconditionerFactory::setFactoryType(dotk::types::right_prec_t type_)
{
    mFactoryType = type_;
}

dotk::types::right_prec_t DOTk_RightPreconditionerFactory::getFactoryType() const
{
    return (mFactoryType);
}

void DOTk_RightPreconditionerFactory::build(const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_template_,
                                            std::tr1::shared_ptr<dotk::DOTk_RightPreconditioner> & right_prec_)
{
}

}
