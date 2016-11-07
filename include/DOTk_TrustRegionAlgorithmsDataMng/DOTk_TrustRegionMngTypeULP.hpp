/*
 * DOTk_TrustRegionMngTypeULP.hpp
 *
 *  Created on: Nov 26, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGIONMNGTYPEULP_HPP_
#define DOTK_TRUSTREGIONMNGTYPEULP_HPP_

#include "DOTk_TrustRegionAlgorithmsDataMng.hpp"

namespace dotk
{

class DOTk_Primal;

template<typename Type>
class vector;
template<typename Type>
class DOTk_ObjectiveFunction;

class DOTk_TrustRegionMngTypeULP : public dotk::DOTk_TrustRegionAlgorithmsDataMng
{
public:
    DOTk_TrustRegionMngTypeULP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                               const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_);
    virtual ~DOTk_TrustRegionMngTypeULP();

    void setForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setParallelForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setParallelCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);
    void setParallelBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::DOTk_Primal> & epsilon_);

private:
    void setFiniteDiffPerturbationVector(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    // unimplemented
    DOTk_TrustRegionMngTypeULP(const dotk::DOTk_TrustRegionMngTypeULP &);
    dotk::DOTk_TrustRegionMngTypeULP operator=(const dotk::DOTk_TrustRegionMngTypeULP &);
};

}

#endif /* DOTK_TRUSTREGIONMNGTYPEULP_HPP_ */
