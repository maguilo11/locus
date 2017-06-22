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
    DOTk_LineSearchMngTypeUNP(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                              const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~DOTk_LineSearchMngTypeUNP();

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
    DOTk_LineSearchMngTypeUNP(const dotk::DOTk_LineSearchMngTypeUNP &);
    dotk::DOTk_LineSearchMngTypeUNP operator=(const dotk::DOTk_LineSearchMngTypeUNP &);
};

}

#endif /* DOTk_LINESEARCHMNGTYPEUNP_HPP_ */
