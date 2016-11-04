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

template<class Type>
class vector;
template<class Type>
class DOTk_ObjectiveFunction;
template<class Type>
class DOTk_EqualityConstraint;

class DOTk_LineSearchMngTypeUNP : public dotk::DOTk_LineSearchAlgorithmsDataMng
{
public:
    DOTk_LineSearchMngTypeUNP(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                              const std::tr1::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_);
    virtual ~DOTk_LineSearchMngTypeUNP();

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
    DOTk_LineSearchMngTypeUNP(const dotk::DOTk_LineSearchMngTypeUNP &);
    dotk::DOTk_LineSearchMngTypeUNP operator=(const dotk::DOTk_LineSearchMngTypeUNP &);
};

}

#endif /* DOTk_LINESEARCHMNGTYPEUNP_HPP_ */
