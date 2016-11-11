/*
 * DOTk_ParallelCentralDiffGrad.hpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PARALLELCENTRALDIFFGRAD_HPP_
#define DOTK_PARALLELCENTRALDIFFGRAD_HPP_

#include <vector>

#include "DOTk_FirstOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;
class DOTk_AssemblyManager;

template<typename Type>
class vector;

class DOTk_ParallelCentralDiffGrad: public dotk::DOTk_FirstOrderOperator
{
public:
    explicit DOTk_ParallelCentralDiffGrad(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_ParallelCentralDiffGrad();

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
    std::tr1::shared_ptr<dotk::vector<Real> > m_FvalPlusEntries;
    std::tr1::shared_ptr<dotk::vector<Real> > m_FvalMinusEntries;
    std::tr1::shared_ptr<dotk::vector<Real> > m_FiniteDiffPerturbationVec;
    std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > m_PerturbedPrimalPlusEntries;
    std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > m_PerturbedPrimalMinusEntries;

private:
    DOTk_ParallelCentralDiffGrad(const dotk::DOTk_ParallelCentralDiffGrad &);
    DOTk_ParallelCentralDiffGrad operator=(const dotk::DOTk_ParallelCentralDiffGrad &);
};

}

#endif /* DOTK_PARALLELCENTRALDIFFGRAD_HPP_ */