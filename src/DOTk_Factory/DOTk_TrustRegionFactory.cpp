/*
 * DOTk_TrustRegionFactory.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "DOTk_Types.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_TrustRegion.hpp"
#include "DOTk_DoglegTrustRegion.hpp"
#include "DOTk_DoubleDoglegTrustRegion.hpp"
#include "DOTk_TrustRegionFactory.hpp"

namespace dotk
{

DOTk_TrustRegionFactory::DOTk_TrustRegionFactory() :
        mTrustRegionType(dotk::types::TRUST_REGION_DISABLED)
{
}

DOTk_TrustRegionFactory::DOTk_TrustRegionFactory(dotk::types::trustregion_t type_) :
        mTrustRegionType(type_)
{
}

DOTk_TrustRegionFactory::~DOTk_TrustRegionFactory()
{
}

dotk::types::trustregion_t DOTk_TrustRegionFactory::getTrustRegionType() const
{
    return (mTrustRegionType);
}

void DOTk_TrustRegionFactory::setWarningMsg(const std::string & msg_)
{
    mWarningMsg.append(msg_);
}

std::string DOTk_TrustRegionFactory::getWarningMsg() const
{
    return (mWarningMsg);
}

void DOTk_TrustRegionFactory::buildCauchyTrustRegion
(std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_method_)
{
    trust_region_method_.reset(new dotk::DOTk_TrustRegion(dotk::types::TRUST_REGION_CAUCHY));
}

void DOTk_TrustRegionFactory::buildDoglegTrustRegion
(std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_method_)
{
    trust_region_method_.reset(new dotk::DOTk_DoglegTrustRegion());
}

void DOTk_TrustRegionFactory::buildDoubleDoglegTrustRegion
(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
 std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_method_)
{
    trust_region_method_.reset(new dotk::DOTk_DoubleDoglegTrustRegion(vector_));
}

void DOTk_TrustRegionFactory::build(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                    std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_step_)
{
    switch(this->getTrustRegionType())
    {
        case dotk::types::TRUST_REGION_CAUCHY:
        {
            trust_region_step_.reset(new dotk::DOTk_TrustRegion(dotk::types::TRUST_REGION_CAUCHY));
            break;
        }
        case dotk::types::TRUST_REGION_DOGLEG:
        {
            trust_region_step_.reset(new dotk::DOTk_DoglegTrustRegion);
            break;
        }
        case dotk::types::TRUST_REGION_DOUBLE_DOGLEG:
        {
            trust_region_step_.reset(new dotk::DOTk_DoubleDoglegTrustRegion(vector_));
            break;
        }
        case dotk::types::TRUST_REGION_DISABLED:
        {
            break;
        }
        default:
        {
            std::ostringstream msg;
            msg << "\nDOTk WARNING: Invalid Trust region method. Default trust region method will be set to "
                    << "Double Dogleg.\n" << std::flush;
            dotk::ioUtils::printMessage(msg);
            this->setWarningMsg(msg.str());
            trust_region_step_.reset(new dotk::DOTk_DoubleDoglegTrustRegion(vector_));
            break;
        }
    }
}

}
