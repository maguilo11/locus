/*
 * DOTk_GmresTestMatrix.hpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_GMRESTESTMATRIX_HPP_
#define DOTK_GMRESTESTMATRIX_HPP_

#include "DOTk_LinearOperator.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;
template<typename ScalarType>
class DOTk_MultiVector;

class DOTk_OptimizationDataMng;

class DOTk_GmresTestMatrix : public dotk::DOTk_LinearOperator
{
public:
    explicit DOTk_GmresTestMatrix(const std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> > & vector_);
    virtual ~DOTk_GmresTestMatrix();

    void apply(const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_,
               const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_);
    void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
               const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_,
               const std::tr1::shared_ptr<dotk::Vector<Real> > & matrix_times_vec_);

private:
    std::tr1::shared_ptr<dotk::matrix<Real> > m_Matrix;

private:
    void allocate(const std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> > & vector_);

private:
    DOTk_GmresTestMatrix(const dotk::DOTk_GmresTestMatrix &);
    dotk::DOTk_GmresTestMatrix & operator=(const dotk::DOTk_GmresTestMatrix &);
};

}

#endif /* DOTK_GMRESTESTMATRIX_HPP_ */
