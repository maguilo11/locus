/*
 * TRROM_BrandLowRankSVD.cpp
 *
 *  Created on: Dec 7, 2016
 *      Author: maguilo
 */

#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"
#include "TRROM_BrandLowRankSVD.hpp"
#include "TRROM_BrandMatrixFactory.hpp"
#include "TRROM_LinearAlgebraFactory.hpp"
#include "TRROM_SpectralDecomposition.hpp"
#include "TRROM_OrthogonalFactorization.hpp"

namespace trrom
{

BrandLowRankSVD::BrandLowRankSVD(const std::tr1::shared_ptr<trrom::BrandMatrixFactory> & brand_factory_,
                                 const std::tr1::shared_ptr<trrom::LinearAlgebraFactory> & linear_algebra_factory_,
                                 const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_,
                                 const std::tr1::shared_ptr<trrom::OrthogonalFactorization> & ortho_) :
        m_BrandFactory(brand_factory_),
        m_SpectralMethod(svd_),
        m_LinearAlgebraFactory(linear_algebra_factory_),
        m_OrthoFactorization(ortho_)
{
}

BrandLowRankSVD::~BrandLowRankSVD()
{
}

void BrandLowRankSVD::solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & data_set_,
                            std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
                            std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
                            std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_)
{
    // Compute \mathbf{M} = \mathbf{U}^{T}\mathbf{Y}\in\mathbb{R}^{{r}\times{k}}
    int num_columns = data_set_->getNumCols();
    int num_rows = left_singular_vectors_->getNumCols();
    std::tr1::shared_ptr<trrom::Matrix<double> > M;
    m_LinearAlgebraFactory->buildLocalMatrix(num_rows, num_columns, M);
    left_singular_vectors_->gemm(true, false, 1., *data_set_, 0., *M);

    // Compute \bar{\mathbf{P}} = \mathbf{Y} - \mathbf{U}\mathbf{M}\in\mathbb{R}^{{m}\times{k}}
    std::tr1::shared_ptr<trrom::Matrix<double> > P_bar = data_set_->create();
    left_singular_vectors_->gemm(false, false, 1., *M, 0., *P_bar);
    P_bar->update(1., *data_set_, -1.);

    // Compute QR factorization of \bar{\mathbf{P}} = \mathbf{Q}\mathbf{R}, where
    // \mathbf{Q}\in\mathbb{R}^{{m}\times{k}} and \mathbf{R}\in\mathbb{R}^{{k}\times{k}}
    std::tr1::shared_ptr<trrom::Matrix<double> > Q;
    std::tr1::shared_ptr<trrom::Matrix<double> > R;
    m_OrthoFactorization->factorize(P_bar, Q, R);

    // Form matrix \mathbf{K}
    std::tr1::shared_ptr<trrom::Matrix<double> > K;
    m_BrandFactory->buildMatrixK(singular_values_, M, R, K);

    // Compute spectral decomposition of \mathbf{K}=\mathbf{C}\mathbf{S}\mathbf{D}^{T},
    // where \mathbf{C},\mathbf{S},\mathbf{D}\in\mathbb{R}^{(r+k)\times(r+k)}
    std::tr1::shared_ptr<trrom::Vector<double> > reduced_singular_values;
    std::tr1::shared_ptr<trrom::Matrix<double> > reduced_left_singular_vectors;
    std::tr1::shared_ptr<trrom::Matrix<double> > reduced_right_singular_vectors;
    m_SpectralMethod->solve(K, reduced_singular_values, reduced_left_singular_vectors, reduced_right_singular_vectors);

    // Set low rank singular value decomposition output
    singular_values_ = reduced_singular_values->create();
    singular_values_->update(1., *reduced_singular_values, 0.);

    // Form matrix \bar{\mathbf{U}}, i.e. updated set of left singular vectors
    std::tr1::shared_ptr<trrom::Matrix<double> > Ubar;
    m_BrandFactory->buildMatrixUbar(left_singular_vectors_, Q, reduced_left_singular_vectors, Ubar);
    left_singular_vectors_ = Ubar->create();
    left_singular_vectors_->update(1., *Ubar, 0.);

    // Form matrix \bar{\mathbf{V}}, i.e. updated set of right singular vectors
    std::tr1::shared_ptr<trrom::Matrix<double> > Vbar;
    m_BrandFactory->buildMatrixVbar(right_singular_vectors_, reduced_right_singular_vectors, Vbar);
    right_singular_vectors_ = Vbar->create();
    right_singular_vectors_->update(1., *Vbar, 0.);
}

}
