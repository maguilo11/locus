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

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class DOTk_ObjectiveFunction;
template<typename ScalarType>
class DOTk_EqualityConstraint;

class DOTk_TrustRegionMngTypeUNP : public dotk::DOTk_TrustRegionAlgorithmsDataMng
{
public:
    DOTk_TrustRegionMngTypeUNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                               const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                               const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~DOTk_TrustRegionMngTypeUNP();

    dotk::types::variable_t getPrimalType() const;

    void setForwardFiniteDiffGradient(const dotk::Vector<Real> & input_);
    void setCentralFiniteDiffGradient(const dotk::Vector<Real> & input_);
    void setBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_);
    void setParallelForwardFiniteDiffGradient(const dotk::Vector<Real> & input_);
    void setParallelCentralFiniteDiffGradient(const dotk::Vector<Real> & input_);
    void setParallelBackwardFiniteDiffGradient(const dotk::Vector<Real> & input_);

private:
    void setFiniteDiffPerturbationVector(const dotk::Vector<Real> & input_);

private:
    // unimplemented
    DOTk_TrustRegionMngTypeUNP(const dotk::DOTk_TrustRegionMngTypeUNP &);
    dotk::DOTk_TrustRegionMngTypeUNP operator=(const dotk::DOTk_TrustRegionMngTypeUNP &);
};

}

#endif /* DOTK_TRUSTREGIONMNGTYPEUNP_HPP_ */
