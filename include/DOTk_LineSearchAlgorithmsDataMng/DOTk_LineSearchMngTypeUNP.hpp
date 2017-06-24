/*
 * DOTk_LineSearchMngTypeUNP.hpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTk_LINESEARCHMNGTYPEUNP_HPP_
#define DOTk_LINESEARCHMNGTYPEUNP_HPP_

#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

class DOTk_Primal;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class DOTk_ObjectiveFunction;
template<typename ScalarType>
class DOTk_EqualityConstraint;

class DOTk_LineSearchMngTypeUNP : public dotk::DOTk_LineSearchAlgorithmsDataMng
{
public:
    DOTk_LineSearchMngTypeUNP(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                              const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & aObjective,
                              const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & aEquality);
    virtual ~DOTk_LineSearchMngTypeUNP();

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
    DOTk_LineSearchMngTypeUNP(const dotk::DOTk_LineSearchMngTypeUNP &);
    dotk::DOTk_LineSearchMngTypeUNP operator=(const dotk::DOTk_LineSearchMngTypeUNP &);
};

}

#endif /* DOTk_LINESEARCHMNGTYPEUNP_HPP_ */
