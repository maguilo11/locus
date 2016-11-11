/*
 * TRROM_TeuchosSVD.cpp
 *
 *  Created on: Oct 19, 2016
 *      Author: maguilo
 */

#include "TRROM_TeuchosSVD.hpp"
#include "TRROM_TeuchosArray.hpp"
#include "TRROM_TeuchosSerialDenseVector.hpp"
#include "TRROM_TeuchosSerialDenseMatrix.hpp"

namespace trrom
{

TeuchosSVD::TeuchosSVD() :
        m_LAPACK()
{
}

TeuchosSVD::~TeuchosSVD()
{
}

void TeuchosSVD::solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & data_,
                       std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
                       std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
                       std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_)
{
    int input_data_num_rows = data_->numRows();
    int input_data_num_columns = data_->numCols();
    int spectral_dimension = std::min(input_data_num_rows, input_data_num_columns);

    // Copy input data since GESVD routine overwrites it (always)
    trrom::TeuchosSerialDenseMatrix<double> matrix(input_data_num_rows, input_data_num_columns);
    matrix.copy(*data_);
    int matrix_leading_dimension = matrix.data()->stride();

    // Resize output data appropriately
    trrom::TeuchosArray<double> singular_values(spectral_dimension);
    trrom::TeuchosSerialDenseMatrix<double> left_singular_vectors(input_data_num_rows, input_data_num_rows);
    trrom::TeuchosSerialDenseMatrix<double> right_singular_vectors(input_data_num_columns, input_data_num_columns);

    int leading_dimension_lsv = left_singular_vectors.data()->stride(); /* lsv = left singular vectors */
    int leading_dimension_rsv = right_singular_vectors.data()->stride(); /* rsv = right singular vectors */

    // Workspace query
    int info;
    int lwork = -1;
    Teuchos::Array<double> work(1);
    // NOTE: 'A' specifies an option for computing all or part of the matrix U
    m_LAPACK.GESVD('A',
                   'A',
                   input_data_num_rows,
                   input_data_num_columns,
                   matrix.data()->values(),
                   matrix_leading_dimension,
                   singular_values.data()->getRawPtr(),
                   left_singular_vectors.data()->values(),
                   leading_dimension_lsv,
                   right_singular_vectors.data()->values(),
                   leading_dimension_rsv,
                   &work[0],
                   lwork,
                   NULL,
                   &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info < 0, std::logic_error, "dgesvd returned info = " << info);

    // Do Singular Value Decomposition (SVD) since no error were detected.
    lwork = work[0];
    work.resize(lwork);
    m_LAPACK.GESVD('A',
                   'A',
                   input_data_num_rows,
                   input_data_num_columns,
                   matrix.data()->values(),
                   matrix_leading_dimension,
                   singular_values.data()->getRawPtr(),
                   left_singular_vectors.data()->values(),
                   leading_dimension_lsv,
                   right_singular_vectors.data()->values(),
                   leading_dimension_rsv,
                   &work[0],
                   lwork,
                   NULL,
                   &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info < 0, std::logic_error, "dgesvd returned info = " << info);

    // Set output data given successful SVD solution
    singular_values_ = singular_values.create();
    singular_values_->copy(singular_values);

    left_singular_vectors_ = left_singular_vectors.create();
    left_singular_vectors_->copy(left_singular_vectors);

    right_singular_vectors_ = right_singular_vectors.create();
    right_singular_vectors_->copy(right_singular_vectors);
}

}
