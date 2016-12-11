/*
 * DOTk_QR.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>

#include "vector.hpp"
#include "DOTk_QR.hpp"
#include "DOTk_MathUtils.hpp"

namespace dotk
{

namespace qr
{

void classicalGramSchmidt(dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_)
{
    int basis_dimension = Q_.basisDimension();

    for(int jth_dim = 0; jth_dim < basis_dimension; ++jth_dim)
    {
        const std::tr1::shared_ptr<dotk::Vector<Real> > & vector = Q_.basis(jth_dim);
        for(int ith_dim = 0; ith_dim <= jth_dim - 1; ++ith_dim)
        {
            Real value = Q_.dot(ith_dim, *vector);
            R_.set(ith_dim, jth_dim, value);
            const std::tr1::shared_ptr<dotk::Vector<Real> > & data = Q_.basis(ith_dim);
            Q_.axpy(jth_dim, -value, *data);
        }
        Real value = Q_.norm(jth_dim);
        R_.set(jth_dim, jth_dim, value);
        value = static_cast<Real>(1.) / value;
        Q_.scale(jth_dim, value);
    }
}

void classicalGramSchmidt(const dotk::matrix<Real> & A_, dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_)
{
    assert(Q_.size() == A_.size());
    Q_.copy(A_);
    R_.fill(0.);
    dotk::qr::classicalGramSchmidt(Q_, R_);
}

void modifiedGramSchmidt(dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_)
{
    int basis_dimension = Q_.basisDimension();

    for(int ith_dim = 0; ith_dim < basis_dimension; ++ith_dim)
    {
        Real value = Q_.norm(ith_dim);
        R_.set(ith_dim, ith_dim, value);
        value = static_cast<Real>(1.) / value;
        Q_.scale(ith_dim, value);
        for(int jth_dim = ith_dim + 1; jth_dim < basis_dimension; ++jth_dim)
        {
            const std::tr1::shared_ptr<dotk::Vector<Real> > & jth_column_data = Q_.basis(jth_dim);
            value = Q_.dot(ith_dim, *jth_column_data);
            R_.set(ith_dim, jth_dim, value);
            const std::tr1::shared_ptr<dotk::Vector<Real> > & ith_column_data = Q_.basis(ith_dim);
            Q_.axpy(jth_dim, -value, *ith_column_data);
        }
    }
}

void modifiedGramSchmidt(const dotk::matrix<Real> & A_, dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_)
{
    assert(Q_.size() == A_.size());
    Q_.copy(A_);
    R_.fill(0.);
    dotk::qr::modifiedGramSchmidt(Q_, R_);
}

void arnoldiModifiedGramSchmidt(const dotk::matrix<Real> & A_,
                                dotk::matrix<Real> & Q_,
                                dotk::matrix<Real> & Hessenberg_,
                                Real tolerance_)
{
    assert(A_.nrows() == Q_.nrows());
    assert(Hessenberg_.ncols() == Q_.nrows());
    assert(Hessenberg_.nrows() == Q_.ncols());

    Q_.fill(0.);
    Hessenberg_.fill(0.);

    Q_.basis(0)->fill(1.);
    Real value = static_cast<Real>(1.) / Q_.basis(0)->norm();
    Q_.scale(value);
    std::tr1::shared_ptr<dotk::Vector<Real> > work = Q_.basis(0)->clone();

    for(size_t dim_index = 0; dim_index < A_.basisDimension(); ++dim_index)
    {
        const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_i = Q_.basis(dim_index);
        A_.matVec(*vector_i, *work);
        for(size_t jth_index = 0; jth_index <= dim_index; ++jth_index)
        {
            const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_j = Q_.basis(jth_index);
            value = work->dot(*vector_j);
            work->update(-value, *vector_j, 1.);
            Hessenberg_(jth_index, dim_index) = value;
        }
        value = work->norm();
        if(value < tolerance_)
        {
            break;
        }
        Hessenberg_(dim_index + 1, dim_index) = value;
        value = static_cast<Real>(1.) / value;
        work->scale(value);
        Q_.basis(dim_index + 1)->update(1., *work, 0.);
        work->fill(0.);
    }
}

void householder(dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_)
{
    Real rho, tau, value;
    size_t basis_dimension = R_.basisDimension();
    std::tr1::shared_ptr<dotk::Vector<Real> > work = Q_.basis(0)->clone();

    for(size_t kth_dim = 0; kth_dim < basis_dimension; ++kth_dim)
    {
        work->update(1., *R_.basis(kth_dim), 0.);
        for(size_t i = 0; i < kth_dim; ++i)
        {
            work->operator[](i) = 0.;
        }
        rho = dotk::sign(work->operator[](kth_dim)) * work->norm();

        work->operator[](kth_dim) = work->operator[](kth_dim) + rho;
        tau = work->operator[](kth_dim) / rho;
        value = static_cast<Real>(1.) / work->operator[](kth_dim);
        work->scale(value);

        for(size_t jth_dim = 0; jth_dim < basis_dimension; ++jth_dim)
        {
            value = tau * work->dot(*R_.basis(jth_dim));
            R_.basis(jth_dim)->update(-value, *work, 1.);
            value = tau * Q_.dot(jth_dim, *work, false);
            Q_.axpy(jth_dim, -value, *work, false);
        }
    }
}

void householder(const dotk::matrix<Real> & A_, dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_)
{
    assert(R_.size() == A_.size());
    assert(R_.basisDimension() == Q_.basisDimension());

    R_.copy(A_);
    Q_.fill(0.);
    size_t basis_dimension = R_.basisDimension();
    for(size_t index = 0; index < basis_dimension; ++index)
    {
        Q_.basis(index)->operator[](index) = 1.;
    }
    dotk::qr::householder(Q_, R_);
}

}

}
