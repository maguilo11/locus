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

template<typename ScalarType>
class DOTk_ObjectiveFunction;
template<typename ScalarType>
class Vector;

class DOTk_TrustRegionMngTypeULP : public dotk::DOTk_TrustRegionAlgorithmsDataMng
{
public:
    DOTk_TrustRegionMngTypeULP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                               const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_);
    virtual ~DOTk_TrustRegionMngTypeULP();

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
    DOTk_TrustRegionMngTypeULP(const dotk::DOTk_TrustRegionMngTypeULP &);
    dotk::DOTk_TrustRegionMngTypeULP operator=(const dotk::DOTk_TrustRegionMngTypeULP &);
};

}

#endif /* DOTK_TRUSTREGIONMNGTYPEULP_HPP_ */
