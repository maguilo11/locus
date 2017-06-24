/*
 * DOTk_LSR1Hessian.hpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LSR1HESSIAN_HPP_
#define DOTK_LSR1HESSIAN_HPP_

#include "DOTk_SecondOrderOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_LSR1Hessian: public dotk::DOTk_SecondOrderOperator
{
public:
    DOTk_LSR1Hessian(const dotk::Vector<Real> & aVector, size_t aSecantStorageSize);
    virtual ~DOTk_LSR1Hessian();

    const std::shared_ptr<dotk::Vector<Real> > & getDeltaGradStorage(size_t aIndex) const;
    const std::shared_ptr<dotk::Vector<Real> > & getDeltaPrimalStorage(size_t aIndex) const;

    void unrollingSR1(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                      const std::shared_ptr<dotk::Vector<Real> > & aOutput);
    void getHessian(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                    const std::shared_ptr<dotk::Vector<Real> > & aOutput);
    virtual void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                       const std::shared_ptr<dotk::Vector<Real> > & aVector,
                       const std::shared_ptr<dotk::Vector<Real> > & aOutput);

private:
    std::shared_ptr<dotk::Vector<Real> > m_DeltaPrimal;
    std::shared_ptr<dotk::Vector<Real> > m_DeltaGradient;

    std::shared_ptr<dotk::matrix<Real> > m_MatrixA;
    std::shared_ptr<dotk::matrix<Real> > m_DeltaPrimalStorage;
    std::shared_ptr<dotk::matrix<Real> > m_DeltaGradientStorage;

private:
    DOTk_LSR1Hessian(const dotk::DOTk_LSR1Hessian &);
    DOTk_LSR1Hessian operator=(const dotk::DOTk_LSR1Hessian &);
};

}

#endif /* DOTK_LSR1HESSIAN_HPP_ */
