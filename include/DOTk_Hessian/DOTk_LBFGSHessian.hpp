/*
 * DOTk_LBFGSHessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LBFGSHESSIAN_HPP_
#define DOTK_LBFGSHESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_LBFGSHessian : public dotk::DOTk_SecondOrderOperator
{
public:
    DOTk_LBFGSHessian(const dotk::Vector<Real> & aVector, size_t aSecantStorageSize);
    virtual ~DOTk_LBFGSHessian();

    const std::shared_ptr<std::vector<Real> > & getDeltaGradPrimalInnerProductStorage() const;
    const std::shared_ptr<dotk::Vector<Real> > & getDeltaGradStorage(size_t at_) const;
    const std::shared_ptr<dotk::Vector<Real> > & getDeltaPrimalStorage(size_t at_) const;

    void getHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                    const std::shared_ptr<dotk::Vector<Real> > & aHessianTimesVector);
    virtual void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng>& mng_,
                       const std::shared_ptr<dotk::Vector<Real> > & aVector,
                       const std::shared_ptr<dotk::Vector<Real> > & aHessianTimesVector);

private:
    std::shared_ptr<std::vector<Real> > m_RhoStorage;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;

    std::shared_ptr<dotk::matrix<Real> > m_MatrixA;
    std::shared_ptr<dotk::matrix<Real> > m_MatrixB;
    std::shared_ptr<dotk::matrix<Real> > m_DeltaPrimalStorage;
    std::shared_ptr<dotk::matrix<Real> > m_DeltaGradientStorage;


private:
    DOTk_LBFGSHessian(const dotk::DOTk_LBFGSHessian &);
    dotk::DOTk_LBFGSHessian operator=(const dotk::DOTk_LBFGSHessian &);
};

}

#endif /* DOTK_LBFGSHESSIAN_HPP_ */
