/*
 * TRROM_LowRankSVD.cpp
 *
 *  Created on: Aug 17, 2016
 */

#include <cassert>

#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"
#include "TRROM_LowRankSVD.hpp"
#include "TRROM_ModifiedGramSchmidt.hpp"

namespace trrom
{

LowRankSVD::LowRankSVD() :
        m_SpectralMethod(),
        m_OrthoFactorization()
{
}

LowRankSVD::LowRankSVD(const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_) :
        m_SpectralMethod(svd_),
        m_OrthoFactorization(new trrom::ModifiedGramSchmidt)
{
}

LowRankSVD::LowRankSVD(const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_,
                       const std::tr1::shared_ptr<trrom::OrthogonalFactorization> & ortho_) :
        m_SpectralMethod(svd_),
        m_OrthoFactorization(ortho_)
{
}

LowRankSVD::~LowRankSVD()
{
}

void LowRankSVD::formMatrixK(const trrom::Vector<double> & singular_values_,
                             const trrom::Matrix<double> & M_,
                             const trrom::Matrix<double> & R_,
                             trrom::Matrix<double> & K_)
{
    /// Form matrix \mathbf{K}\in\mathbb{R}^{(r+k)\times(r+k)}
    /// Parameters:
    ///     singular_values_ - (In) previous (k-1) singular values, where k is the optimization iteration counter
    ///     M_               - (In) \mathbf{M} = \mathbf{U}^{T}\mathbf{Y}\in\mathbb{R}^{{r}\times{k}},
    ///                        where \mathbf{U} is the set of left singular vectors, \mathbf{Y} is the new set
    ///                        of snapshots, r = rank (# singular values), and k = # of new snapshots
    ///     R_               - (In) upper triangular matrix computed from a QR decomposition, where
    ///                        \mathbf{R}\in\mathbb{R}^{k\times{k}}
    ///     K_               - (Out) \mathbf{K}=\left[\Sigma \mathbf{M}; 0 \mathbf{R}\right]
    assert(M_.getNumCols() == R_.getNumCols());
    assert(R_.getNumRows() == R_.getNumCols());
    assert(singular_values_.size() == M_.getNumRows());
    assert(K_.getNumRows() == (singular_values_.size() + M_.getNumCols()));
    assert(K_.getNumCols() == (singular_values_.size() + M_.getNumCols()));

    int num_new_snapshots = M_.getNumCols();
    int num_singular_values = singular_values_.size();
    for(int index = 0; index < num_singular_values; ++index)
    {
        double value = singular_values_[index];
        K_(index, index) = value;
        for(int j = 0; j < num_new_snapshots; ++j)
        {
            int j_column = num_singular_values + j;
            value = M_(index, j);
            K_(index, j_column) = value;
            int k_row = num_singular_values + j;
            for(int k = 0; k < num_new_snapshots; ++k)
            {
                int k_column = num_singular_values + k;
                value = R_(j, k);
                K_(k_row, k_column) = value;
            }
        }
    }
}

void LowRankSVD::formMatrixUbar(const trrom::Matrix<double> & left_singular_vectors_,
                                const trrom::Matrix<double> & Q_,
                                const trrom::Matrix<double> & C_,
                                trrom::Matrix<double> & Ubar_)
{
    /// Form matrix \bar{\mathbf{U}}\in\mathbb{R}^{m\times(r+k)}, where m = # dofs,
    /// r = rank (# singular values), and k = # new snapshots
    /// Parameters:
    ///     left_singular_vectors_ - (In) previous (k-1) left singular vectors, where k is the optimization iteration
    ///                              counter, where \Phi\in\mathbb{m\times{r}}
    ///     Q_                     - (In) \mathbf{Q}\in\mathbb{R}^{m\times{k}} denotes the orthonormal matrix obtained
    ///                              from a QR decomposition
    ///     C_                     - (In) left singular values obtained from a singular value decompostion of \mathbf{K}
    ///                              =\left[\Sigma \mathbf{M}; 0 \mathbf{R}\right], see description for function formMatrixK
    ///     Ubar_                  - (Out) new left singular basis \bar{\mathbf{U}} = \left[\Phi \mathbf{Q}\right]\mathbf{C}
    assert(Ubar_.getNumCols() == C_.getNumCols());
    assert(left_singular_vectors_.getNumRows() == Q_.getNumRows());
    assert(left_singular_vectors_.getNumRows() == Ubar_.getNumRows());
    assert(C_.getNumRows() == (Q_.getNumCols() + left_singular_vectors_.getNumCols()));
    assert(C_.getNumCols() == (Q_.getNumCols() + left_singular_vectors_.getNumCols()));

    int num_new_snapshots = Q_.getNumCols();
    int new_singular_vectors = C_.getNumCols();
    int num_dofs = left_singular_vectors_.getNumRows();
    int num_singular_vectors = left_singular_vectors_.getNumCols();
    std::tr1::shared_ptr<trrom::Matrix<double> > matrix = left_singular_vectors_.create(num_dofs, new_singular_vectors);
    for(int row = 0; row < num_dofs; ++row)
    {
        for(int j = 0; j < num_singular_vectors; ++j)
        {
            double value = left_singular_vectors_(row, j);
            (*matrix)(row, j) = value;
        }
        for(int k = 0; k < num_new_snapshots; ++k)
        {
            int column = num_singular_vectors + k;
            double value = Q_(row, k);
            (*matrix)(row, column) = value;
        }
    }
    matrix->gemm(false, false, 1., C_, 0., Ubar_);
}

void LowRankSVD::formMatrixVbar(int num_new_snapshots_,
                                const trrom::Matrix<double> & right_singular_vectors_,
                                const trrom::Matrix<double> & D_,
                                trrom::Matrix<double> & Vbar_)
{
    /// Form matrix \bar{\mathbf{V}}\in\mathbb{R}^{n\times{r}}, where n = # old snapshots
    /// and r = rank (# singular values)
    /// Parameters:
    ///     num_new_snapshots_      - (In) number of new snapshots
    ///     right_singular_vectors_ - (In) previous (k-1) right singular vectors, where k is the optimization iteration
    ///                               counter, where \Psi\in\mathbb{n\times{r}}
    ///     D_                      - (In) right singular values obtained from a singular value decompostion of \mathbf{K}
    ///                              =\left[\Sigma \mathbf{M}; 0 \mathbf{R}\right], see description for function formMatrixK
    ///     Vbar_                   - (Out) new right singular basis \bar{\mathbf{V}} = \left[\mathbf{V} 0; 0 \mathbf{I}\right]\mathbf{D}
    assert(num_new_snapshots_ > 0);
    assert(Vbar_.getNumCols() == D_.getNumCols());
    assert(D_.getNumRows() == (num_new_snapshots_ + right_singular_vectors_.getNumCols()));
    assert(D_.getNumCols() == (num_new_snapshots_ + right_singular_vectors_.getNumCols()));
    assert(right_singular_vectors_.getNumRows() == (Vbar_.getNumRows() - num_new_snapshots_));
    assert(right_singular_vectors_.getNumCols() == (Vbar_.getNumCols() - num_new_snapshots_));
    assert(Vbar_.getNumRows() == (right_singular_vectors_.getNumRows() + num_new_snapshots_));

    int spectral_dimension = right_singular_vectors_.getNumCols();
    int num_old_snapshots = right_singular_vectors_.getNumRows();
    std::tr1::shared_ptr<trrom::Matrix<double> > matrix = Vbar_.create(Vbar_.getNumRows(), Vbar_.getNumCols());
    for(int i = 0; i < num_old_snapshots; ++i)
    {
        for(int j = 0; j < spectral_dimension; ++j)
        {
            double value = right_singular_vectors_(i, j);
            (*matrix)(i, j) = value;
        }
    }
    // Set Block 22 = \mathbf{I}
    for(int index = 0; index < num_new_snapshots_; ++index)
    {
        int row = num_old_snapshots + index;
        int column = spectral_dimension + index;
        (*matrix)(row, column) = 1.;
    }
    matrix->gemm(false, false, 1., D_, 0., Vbar_);
}

void LowRankSVD::solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & new_snapshots_,
                       std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
                       std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
                       std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_)
{
    // Compute \mathbf{M} = \mathbf{U}^{T}\mathbf{Y}\in\mathbb{R}^{{r}\times{k}}
    int num_snapshots = new_snapshots_->getNumCols();
    int num_singular_vectors = left_singular_vectors_->getNumCols();
    std::tr1::shared_ptr<trrom::Matrix<double> > M = new_snapshots_->create(num_singular_vectors, num_snapshots);
    left_singular_vectors_->gemm(true, false, 1., *new_snapshots_, 0., *M);

    // Compute \bar{\mathbf{P}} = \mathbf{Y} - \mathbf{U}\mathbf{M}\in\mathbb{R}^{{m}\times{k}}
    int num_dofs = new_snapshots_->getNumRows();
    std::tr1::shared_ptr<trrom::Matrix<double> > Pb = new_snapshots_->create(num_dofs, num_snapshots);
    left_singular_vectors_->gemm(false, false, -1., *M, 0., *Pb);
    Pb->update(1., *new_snapshots_, 1.);

    // Compute QR factorization of \bar{\mathbf{P}} = \mathbf{Q}\mathbf{R}, where
    // \mathbf{Q}\in\mathbb{R}^{{m}\times{k}} and \mathbf{R}\in\mathbb{R}^{{k}\times{k}}
    std::tr1::shared_ptr<trrom::Matrix<double> > Q = new_snapshots_->create(num_dofs, num_snapshots);
    std::tr1::shared_ptr<trrom::Matrix<double> > R = new_snapshots_->create(num_snapshots, num_snapshots);
    m_OrthoFactorization->factorize(*Pb, *Q, *R);

    // Form matrix \mathbf{K}
    int num_rows = num_singular_vectors + R->getNumRows();
    int num_columns = num_singular_vectors + num_snapshots;
    std::tr1::shared_ptr<trrom::Matrix<double> > K = left_singular_vectors_->create(num_rows, num_columns);
    this->formMatrixK(*singular_values_, *M, *R, *K);

    // Compute spectral decomposition of \mathbf{K}=\mathbf{C}\mathbf{S}\mathbf{D}^{T},
    // where \mathbf{C},\mathbf{S},\mathbf{D}\in\mathbb{R}^{(r+k)\times(r+k)}
    std::tr1::shared_ptr<trrom::Matrix<double> > left_singular_vectors;
    std::tr1::shared_ptr<trrom::Matrix<double> > right_singular_vectors;
    std::tr1::shared_ptr<trrom::Vector<double> > singular_values = singular_values_->create();
    m_SpectralMethod->solve(K, singular_values, left_singular_vectors, right_singular_vectors);

    // Set low rank singular value decomposition output
    singular_values_ = singular_values->create();
    singular_values_->update(1., *singular_values, 0.);

    // Form matrix \bar{\mathbf{U}}, i.e. updated set of left singular vectors
    num_columns = left_singular_vectors->getNumCols();
    std::tr1::shared_ptr<trrom::Matrix<double> > Ubar = new_snapshots_->create(num_dofs, num_columns);
    this->formMatrixUbar(*left_singular_vectors_, *Q, *left_singular_vectors, *Ubar);
    left_singular_vectors_ = Ubar->create();
    left_singular_vectors_->update(1., *Ubar, 0.);

    // Form matrix \bar{\mathbf{V}}, i.e. updated set of right singular vectors
    num_columns = right_singular_vectors->getNumRows();
    int dim = right_singular_vectors->getNumRows() - left_singular_vectors_->getNumCols();
    num_rows = left_singular_vectors_->getNumRows() + dim;
    std::tr1::shared_ptr<trrom::Matrix<double> > Vbar = new_snapshots_->create(num_rows, num_columns);
    this->formMatrixVbar(num_snapshots, *right_singular_vectors_, *right_singular_vectors, *Vbar);
    right_singular_vectors_ = Vbar->create();
    right_singular_vectors_->update(1., *Vbar, 0.);
}

}
