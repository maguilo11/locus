/*
 * DOTk_ParallelForwardDiffGrad.hpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PARALLELFORWARDDIFFGRAD_HPP_
#define DOTK_PARALLELFORWARDDIFFGRAD_HPP_

#include <vector>

#include "DOTk_FirstOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;
class DOTk_AssemblyManager;

template<typename ScalarType>
class Vector;

class DOTk_ParallelForwardDiffGrad : public dotk::DOTk_FirstOrderOperator
{
public:
    explicit DOTk_ParallelForwardDiffGrad(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_);
    virtual ~DOTk_ParallelForwardDiffGrad();

    const std::tr1::shared_ptr<dotk::Vector<Real> > & getFiniteDiffPerturbationVec() const;
    virtual void setFiniteDiffPerturbationVec(const dotk::Vector<Real> & input_);

    void getGradient(const Real & fval_,
                     const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & interface_,
                     const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                     const std::tr1::shared_ptr<dotk::Vector<Real> > & grad_);
    virtual void gradient(const dotk::DOTk_OptimizationDataMng * const mng_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_Fval;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_FiniteDiffPerturbationVec;
    std::vector<std::tr1::shared_ptr<dotk::Vector<Real> > > m_PerturbedPrimal;

private:
    DOTk_ParallelForwardDiffGrad(const dotk::DOTk_ParallelForwardDiffGrad &);
    DOTk_ParallelForwardDiffGrad operator=(const dotk::DOTk_ParallelForwardDiffGrad &);
};

}

#endif /* DOTK_PARALLELFORWARDDIFFGRAD_HPP_ */
