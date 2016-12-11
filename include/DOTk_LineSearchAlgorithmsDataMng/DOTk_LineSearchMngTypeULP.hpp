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
class matrix;
template<typename ScalarType>
class DOTk_ObjectiveFunction;

class DOTk_LineSearchMngTypeULP : public dotk::DOTk_LineSearchAlgorithmsDataMng
{
public:
    explicit DOTk_LineSearchMngTypeULP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    DOTk_LineSearchMngTypeULP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & operators_);
    virtual ~DOTk_LineSearchMngTypeULP();

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
    DOTk_LineSearchMngTypeULP(const dotk::DOTk_LineSearchMngTypeULP &);
    DOTk_LineSearchMngTypeULP operator=(const dotk::DOTk_LineSearchMngTypeULP &);
};

}

#endif /* DOTK_LINESEARCHMNGTYPEULP_HPP_ */
