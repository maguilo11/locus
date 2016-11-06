/*
 * DOTk_ParallelBackwardDiffGrad.hpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PARALLELBACKWARDDIFFGRAD_HPP_
#define DOTK_PARALLELBACKWARDDIFFGRAD_HPP_

#include <vector>

#include "DOTk_FirstOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;
class DOTk_AssemblyManager;

template<typename Type>
class vector;

class DOTk_ParallelBackwardDiffGrad: public dotk::DOTk_FirstOrderOperator
{
public:
    explicit DOTk_ParallelBackwardDiffGrad(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_ParallelBackwardDiffGrad();

    const std::tr1::shared_ptr<dotk::vector<Real> > & getFiniteDiffPerturbationVec() const;
    virtual void setFiniteDiffPerturbationVec(const dotk::vector<Real> & input_);

    void getGradient(Real fval_,
                     const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & interface_,
                     const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                     const std::tr1::shared_ptr<dotk::vector<Real> > & grad_);
    virtual void gradient(const dotk::DOTk_OptimizationDataMng * const mng_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_Fval;
    std::tr1::shared_ptr<dotk::vector<Real> > m_FiniteDiffPerturbationVec;
    std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > m_PerturbedPrimal;

private:
    DOTk_ParallelBackwardDiffGrad(const dotk::DOTk_ParallelBackwardDiffGrad &);
    DOTk_ParallelBackwardDiffGrad operator=(const dotk::DOTk_ParallelBackwardDiffGrad &);
};

}

#endif /* DOTK_PARALLELBACKWARDDIFFGRAD_HPP_ */
