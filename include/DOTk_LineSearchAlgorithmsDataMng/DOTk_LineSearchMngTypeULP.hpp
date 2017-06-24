/*
 * DOTk_LineSearchUnconstrainedMng.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCHMNGTYPEULP_HPP_
#define DOTK_LINESEARCHMNGTYPEULP_HPP_

#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

class DOTk_Primal;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class DOTk_ObjectiveFunction;

class DOTk_LineSearchMngTypeULP : public dotk::DOTk_LineSearchAlgorithmsDataMng
{
public:
    explicit DOTk_LineSearchMngTypeULP(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);
    DOTk_LineSearchMngTypeULP(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                              const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & aLinearOperator);
    virtual ~DOTk_LineSearchMngTypeULP();

    void setForwardFiniteDiffGradient(const dotk::Vector<Real> & aInput);
    void setCentralFiniteDiffGradient(const dotk::Vector<Real> & aInput);
    void setBackwardFiniteDiffGradient(const dotk::Vector<Real> & aInput);
    void setParallelForwardFiniteDiffGradient(const dotk::Vector<Real> & aInput);
    void setParallelCentralFiniteDiffGradient(const dotk::Vector<Real> & aInput);
    void setParallelBackwardFiniteDiffGradient(const dotk::Vector<Real> & aInput);

private:
    void setFiniteDiffPerturbationVector(const dotk::Vector<Real> & aInput);

private:
    // unimplemented
    DOTk_LineSearchMngTypeULP(const dotk::DOTk_LineSearchMngTypeULP &);
    DOTk_LineSearchMngTypeULP operator=(const dotk::DOTk_LineSearchMngTypeULP &);
};

}

#endif /* DOTK_LINESEARCHMNGTYPEULP_HPP_ */
