/*
 * DOTk_CentralDifferenceGrad.hpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_CENTRALDIFFERENCEGRAD_HPP_
#define DOTK_CENTRALDIFFERENCEGRAD_HPP_

#include "DOTk_FirstOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;
class DOTk_AssemblyManager;

template<class Type>
class vector;

class DOTk_CentralDifferenceGrad : public dotk::DOTk_FirstOrderOperator
{
public:
    explicit DOTk_CentralDifferenceGrad(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_CentralDifferenceGrad();

    const std::tr1::shared_ptr<dotk::vector<Real> > & getFiniteDiffPerturbationVec() const;
    virtual void setFiniteDiffPerturbationVec(const dotk::vector<Real> & input_);

    void getGradient(const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & interface_,
                     const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                     const std::tr1::shared_ptr<dotk::vector<Real> > & grad_);
    virtual void gradient(const dotk::DOTk_OptimizationDataMng * const mng_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_FiniteDiffPerturbationVec;

private:
    DOTk_CentralDifferenceGrad(const dotk::DOTk_CentralDifferenceGrad &);
    DOTk_CentralDifferenceGrad operator=(const dotk::DOTk_CentralDifferenceGrad &);
};

}

#endif /* DOTK_CENTRALDIFFERENCEGRAD_HPP_ */