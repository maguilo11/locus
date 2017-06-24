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
    explicit DOTk_GmresTestMatrix(const std::shared_ptr<dotk::DOTk_MultiVector<Real> > & aVector);
    virtual ~DOTk_GmresTestMatrix();

    void apply(const std::shared_ptr<dotk::Vector<Real> > & aVector,
               const std::shared_ptr<dotk::Vector<Real> > & aMatrixTimesVector);
    void apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
               const std::shared_ptr<dotk::Vector<Real> > & aVector,
               const std::shared_ptr<dotk::Vector<Real> > & aMatrixTimesVector);

private:
    size_t mNumRows;
    std::shared_ptr<dotk::matrix<Real> > mMatrix;

private:
    void allocate(const std::shared_ptr<dotk::DOTk_MultiVector<Real> > & aVector);

private:
    DOTk_GmresTestMatrix(const dotk::DOTk_GmresTestMatrix &);
    dotk::DOTk_GmresTestMatrix & operator=(const dotk::DOTk_GmresTestMatrix &);
};

}

#endif /* DOTK_GMRESTESTMATRIX_HPP_ */
