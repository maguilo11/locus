/*
 * DOTk_TrustRegionFactory.hpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGIONFACTORY_HPP_
#define DOTK_TRUSTREGIONFACTORY_HPP_

#include <string>
#include <memory>

#include <DOTk_Types.hpp>

namespace dotk
{

class DOTk_TrustRegion;

template<typename ScalarType>
class Vector;

class DOTk_TrustRegionFactory
{
public:
    DOTk_TrustRegionFactory();
    explicit DOTk_TrustRegionFactory(dotk::types::trustregion_t type_);
    ~DOTk_TrustRegionFactory();

    dotk::types::trustregion_t getTrustRegionType() const;
    void setWarningMsg(const std::string & msg_);
    std::string getWarningMsg() const;

    void buildCauchyTrustRegion(std::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_method_);
    void buildDoglegTrustRegion(std::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_method_);
    void buildDoubleDoglegTrustRegion(const std::shared_ptr<dotk::Vector<Real> > & vector_,
                                      std::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_method_);
    void build(const std::shared_ptr<dotk::Vector<Real> > & vector_,
               std::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_step_);

private:
    std::string mWarningMsg;
    dotk::types::trustregion_t mTrustRegionType;

private:
    DOTk_TrustRegionFactory(const dotk::DOTk_TrustRegionFactory&);
    dotk::DOTk_TrustRegionFactory operator=(const dotk::DOTk_TrustRegionFactory&);
};

}

#endif /* DOTK_TRUSTREGIONFACTORY_HPP_ */
