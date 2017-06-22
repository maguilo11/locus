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
    explicit DOTk_LineSearchMngTypeULP(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    DOTk_LineSearchMngTypeULP(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_);
    virtual ~DOTk_LineSearchMngTypeULP();

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
    DOTk_LineSearchMngTypeULP(const dotk::DOTk_LineSearchMngTypeULP &);
    DOTk_LineSearchMngTypeULP operator=(const dotk::DOTk_LineSearchMngTypeULP &);
};

}

#endif /* DOTK_LINESEARCHMNGTYPEULP_HPP_ */
