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

DOTk_TrustRegionFactory::DOTk_TrustRegionFactory(dotk::types::trustregion_t aType) :
        mTrustRegionType(aType)
{
}

DOTk_TrustRegionFactory::~DOTk_TrustRegionFactory()
{
}

dotk::types::trustregion_t DOTk_TrustRegionFactory::getTrustRegionType() const
{
    return (mTrustRegionType);
}

void DOTk_TrustRegionFactory::setWarningMsg(const std::string & aMsg)
{
    mWarningMsg.append(aMsg);
}

std::string DOTk_TrustRegionFactory::getWarningMsg() const
{
    return (mWarningMsg);
}

void DOTk_TrustRegionFactory::buildCauchyTrustRegion
(std::shared_ptr<dotk::DOTk_TrustRegion> & aOutput)
{
    dotk::types::trustregion_t tType = dotk::types::TRUST_REGION_CAUCHY;
    aOutput = std::make_shared<dotk::DOTk_TrustRegion>(tType);
}

void DOTk_TrustRegionFactory::buildDoglegTrustRegion
(std::shared_ptr<dotk::DOTk_TrustRegion> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_DoglegTrustRegion>();
}

void DOTk_TrustRegionFactory::buildDoubleDoglegTrustRegion
(const std::shared_ptr<dotk::Vector<Real> > & aVector,
 std::shared_ptr<dotk::DOTk_TrustRegion> & aOutput)
{
    aOutput = std::make_shared<dotk::DOTk_DoubleDoglegTrustRegion>(aVector);
}

void DOTk_TrustRegionFactory::build(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                    std::shared_ptr<dotk::DOTk_TrustRegion> & aOutput)
{
    switch(this->getTrustRegionType())
    {
        case dotk::types::TRUST_REGION_CAUCHY:
        {
            dotk::types::trustregion_t tType = dotk::types::TRUST_REGION_CAUCHY;
            aOutput = std::make_shared<dotk::DOTk_TrustRegion>(tType);
            break;
        }
        case dotk::types::TRUST_REGION_DOGLEG:
        {
            aOutput = std::make_shared<dotk::DOTk_DoglegTrustRegion>();
            break;
        }
        case dotk::types::TRUST_REGION_DOUBLE_DOGLEG:
        {
            aOutput = std::make_shared<dotk::DOTk_DoubleDoglegTrustRegion>(aVector);
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
            aOutput = std::make_shared<dotk::DOTk_DoubleDoglegTrustRegion>(aVector);
            break;
        }
    }
}

}
