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
    assert(M_.numCols() == R_.numCols());
    assert(R_.numRows() == R_.numCols());
    assert(singular_values_.size() == M_.numRows());
    assert(K_.numRows() == (singular_values_.size() + M_.numCols()));
    assert(K_.numCols() == (singular_values_.size() + M_.numCols()));

    int num_new_snapshots = M_.numCols();
    int num_singular_values = singular_values_.size();
    for(int index = 0; index < num_singular_values; ++index)
    {
        K_(index, index) = singular_values_[index];
        for(int j = 0; j < num_new_snapshots; ++j)
        {
            int j_column = num_singular_values + j;
            K_(index, j_column) = M_(index, j);
            int k_row = num_singular_values + j;
            for(int k = 0; k < num_new_snapshots; ++k)
            {
                int k_column = num_singular_values + k;
                K_(k_row, k_column) = R_(j, k);
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
    assert(Ubar_.numCols() == C_.numCols());
    assert(left_singular_vectors_.numRows() == Q_.numRows());
    assert(left_singular_vectors_.numRows() == Ubar_.numRows());
    assert(C_.numRows() == (Q_.numCols() + left_singular_vectors_.numCols()));
    assert(C_.numCols() == (Q_.numCols() + left_singular_vectors_.numCols()));

    int num_new_snapshots = Q_.numCols();
    int new_singular_vectors = C_.numCols();
    int num_dofs = left_singular_vectors_.numRows();
    int num_singular_vectors = left_singular_vectors_.numCols();
    std::tr1::shared_ptr<trrom::Matrix<double> > matrix = left_singular_vectors_.create(num_dofs, new_singular_vectors);
    for(int row = 0; row < num_dofs; ++row)
    {
        for(int j = 0; j < num_singular_vectors; ++j)
        {
            (*matrix)(row, j) = left_singular_vectors_(row, j);
        }
        for(int k = 0; k < num_new_snapshots; ++k)
        {
            int column = num_singular_vectors + k;
            (*matrix)(row, column) = Q_(row, k);
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
    assert(Vbar_.numCols() == D_.numCols());
    assert(D_.numRows() == (num_new_snapshots_ + right_singular_vectors_.numCols()));
    assert(D_.numCols() == (num_new_snapshots_ + right_singular_vectors_.numCols()));
    assert(right_singular_vectors_.numRows() == (Vbar_.numRows() - num_new_snapshots_));
    assert(right_singular_vectors_.numCols() == (Vbar_.numCols() - num_new_snapshots_));
    assert(Vbar_.numRows() == (right_singular_vectors_.numRows() + num_new_snapshots_));

    int rank = right_singular_vectors_.numCols();
    int num_old_snapshots = right_singular_vectors_.numRows();
    std::tr1::shared_ptr<trrom::Matrix<double> > matrix = Vbar_.create(Vbar_.numRows(), Vbar_.numCols());
    for(int i = 0; i < num_old_snapshots; ++i)
    {
        for(int j = 0; j < rank; ++j)
        {
            (*matrix)(i, j) = right_singular_vectors_(i, j);
        }
    }
    // Set Block 22 = \mathbf{I}
    for(int i = 0; i < num_new_snapshots_; ++i)
    {
        int row = num_old_snapshots + i;
        int column = rank + i;
        (*matrix)(row, column) = static_cast<double>(1.);
    }
    matrix->gemm(false, false, 1., D_, 0., Vbar_);
}

void LowRankSVD::solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & new_snapshots_,
                       std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
                       std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
                       std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_)
{
    // Compute \mathbf{M} = \mathbf{U}^{T}\mathbf{Y}\in\mathbb{R}^{{r}\times{k}}
    int num_snapshots = new_snapshots_->numCols();
    int num_singular_vectors = left_singular_vectors_->numCols();
    std::tr1::shared_ptr<trrom::Matrix<double> > M = new_snapshots_->create(num_singular_vectors, num_snapshots);
    left_singular_vectors_->gemm(true, false, 1., *new_snapshots_, 0., *M);

    // Compute \bar{\mathbf{P}} = \mathbf{Y} - \mathbf{U}\mathbf{M}\in\mathbb{R}^{{m}\times{k}}
    int num_dofs = new_snapshots_->numRows();
    std::tr1::shared_ptr<trrom::Matrix<double> > Pb = new_snapshots_->create(num_dofs, num_snapshots);
    left_singular_vectors_->gemm(false, false, -1., *M, 0., *Pb);
    Pb->add(1., *new_snapshots_);

    // Compute QR factorization of \bar{\mathbf{P}} = \mathbf{Q}\mathbf{R}, where
    // \mathbf{Q}\in\mathbb{R}^{{m}\times{k}} and \mathbf{R}\in\mathbb{R}^{{k}\times{k}}
    std::tr1::shared_ptr<trrom::Matrix<double> > Q = new_snapshots_->create(num_dofs, num_snapshots);
    std::tr1::shared_ptr<trrom::Matrix<double> > R = new_snapshots_->create(num_snapshots, num_snapshots);
    m_OrthoFactorization->factorize(*Pb, *Q, *R);

    // Form matrix \mathbf{K}
    int numRows = num_singular_vectors + R->numRows();
    int numCols = num_singular_vectors + num_snapshots;
    std::tr1::shared_ptr<trrom::Matrix<double> > K = left_singular_vectors_->create(numRows, numCols);
    this->formMatrixK(*singular_values_, *M, *R, *K);

    // Compute spectral decomposition of \mathbf{K}=\mathbf{C}\mathbf{S}\mathbf{D}^{T},
    // where \mathbf{C},\mathbf{S},\mathbf{D}\in\mathbb{R}^{(r+k)\times(r+k)}
    std::tr1::shared_ptr<trrom::Matrix<double> > left_singular_vectors;
    std::tr1::shared_ptr<trrom::Matrix<double> > right_singular_vectors;
    std::tr1::shared_ptr<trrom::Vector<double> > singular_values = singular_values_->create();
    m_SpectralMethod->solve(K, singular_values, left_singular_vectors, right_singular_vectors);

    // Set low rank singular value decomposition output
    singular_values_ = singular_values->create();
    singular_values_->copy(*singular_values);

    // Form matrix \bar{\mathbf{U}}, i.e. updated set of left singular vectors
    numCols = left_singular_vectors->numCols();
    std::tr1::shared_ptr<trrom::Matrix<double> > Ubar = new_snapshots_->create(num_dofs, numCols);
    this->formMatrixUbar(*left_singular_vectors_, *Q, *left_singular_vectors, *Ubar);
    left_singular_vectors_ = Ubar->create();
    left_singular_vectors_->copy(*Ubar);

    // Form matrix \bar{\mathbf{V}}, i.e. updated set of right singular vectors
    numCols = right_singular_vectors->numRows();
    int dim = right_singular_vectors->numRows() - left_singular_vectors_->numCols();
    numRows = left_singular_vectors_->numRows() + dim;
    std::tr1::shared_ptr<trrom::Matrix<double> > Vbar = new_snapshots_->create(numRows, numCols);
    this->formMatrixVbar(num_snapshots, *right_singular_vectors_, *right_singular_vectors, *Vbar);
    right_singular_vectors_ = Vbar->create();
    right_singular_vectors_->copy(*Vbar);
}

}
