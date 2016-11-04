/*
 * DOTk_TrustRegionMngTypeUNP.hpp
 *
 *  Created on: Apr 24, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGIONMNGTYPEUNP_HPP_
#define DOTK_TRUSTREGIONMNGTYPEUNP_HPP_

#include "DOTk_TrustRegionAlgorithmsDataMng.hpp"

namespace dotk
{

class DOTk_Primal;

template<class Type>
class vector;
template<class Type>
class DOTk_ObjectiveFunction;
template<class Type>
class DOTk_EqualityConstraint;

class DOTk_TrustRegionMngTypeUNP : public dotk::DOTk_TrustRegionAlgorithmsDataMng
{
public:
    DOTk_TrustRegionMngTypeUNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                               const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                               const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~DOTk_TrustRegionMngTypeUNP();

    dotk::types::variable_t getPrimalType() const;

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
    DOTk_TrustRegionMngTypeUNP(const dotk::DOTk_TrustRegionMngTypeUNP &);
    dotk::DOTk_TrustRegionMngTypeUNP operator=(const dotk::DOTk_TrustRegionMngTypeUNP &);
};

}

#endif /* DOTK_TRUSTREGIONMNGTYPEUNP_HPP_ */
