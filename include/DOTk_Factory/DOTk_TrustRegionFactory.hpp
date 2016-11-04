/*
 * DOTk_TrustRegionFactory.hpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGIONFACTORY_HPP_
#define DOTK_TRUSTREGIONFACTORY_HPP_

namespace dotk
{

class DOTk_TrustRegion;

class DOTk_TrustRegionFactory
{
public:
    DOTk_TrustRegionFactory();
    explicit DOTk_TrustRegionFactory(dotk::types::trustregion_t type_);
    ~DOTk_TrustRegionFactory();

    dotk::types::trustregion_t getTrustRegionType() const;
    void setWarningMsg(const std::string & msg_);
    std::string getWarningMsg() const;

    void buildCauchyTrustRegion(std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_method_);
    void buildDoglegTrustRegion(std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_method_);
    void buildDoubleDoglegTrustRegion(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                      std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_method_);
    void build(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
               std::tr1::shared_ptr<dotk::DOTk_TrustRegion> & trust_region_step_);

private:
    std::string mWarningMsg;
    dotk::types::trustregion_t mTrustRegionType;

private:
    DOTk_TrustRegionFactory(const dotk::DOTk_TrustRegionFactory&);
    dotk::DOTk_TrustRegionFactory operator=(const dotk::DOTk_TrustRegionFactory&);
};

}

#endif /* DOTK_TRUSTREGIONFACTORY_HPP_ */
