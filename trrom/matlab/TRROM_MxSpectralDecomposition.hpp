/*
 * TRROM_MxSpectralDecomposition.hpp
 *
 *  Created on: Nov 12, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_MATLAB_INCLUDE_TRROM_MXSPECTRALDECOMPOSITION_HPP_
#define TRROM_MATLAB_INCLUDE_TRROM_MXSPECTRALDECOMPOSITION_HPP_

#include "mex.h"
#include "lapack.h"
#include "TRROM_MxVector.hpp"

namespace trrom
{

template<typename ScalarType>
class Matrix;

class MxSingularValueDecomposition : public trrom::SpectralDecomposition
{
public:
    MxSingularValueDecomposition()
    {
    }
    virtual ~MxSingularValueDecomposition()
    {
    }

    void solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & data_,
               std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
               std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
               std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_)
    {
        /* duplicate input matrix (since its contents will be overwritten) */
        mxArray *in = mxDuplicateArray(prhs[0]);

        /* dimensions of input matrix */
        mwSignedIndex m = mxGetM(in);
        mwSignedIndex n = mxGetN(in);

        /* create output matrices */
        int spectral_dimension = (m < n) ? m : n;
        plhs[0] = mxCreateDoubleMatrix(m, m, mxREAL);
        plhs[2] = mxCreateDoubleMatrix(n, n, mxREAL);
        trrom::MxVector singular_values(spectral_dimension);

        /* get pointers to data */
        double* data_matrix = mxGetPr(in);
        double* left_singular_vectors = mxGetPr(plhs[0]);
        double* right_singular_vectors = mxGetPr(plhs[2]);

        /* query and allocate the optimal workspace size */
        double workopt = 0;
        mwSignedIndex info = 0;
        mwSignedIndex lwork = -1;
        dgesvd("A",
               "A",
               &m,
               &n,
               data_matrix,
               &m,
               singular_values.data(),
               left_singular_vectors,
               &m,
               right_singular_vectors,
               &n,
               &workopt,
               &lwork,
               &info);
        lwork = static_cast<mwSignedIndex>(workopt);
        double* work = (double *) mxMalloc(lwork * sizeof(double));

        /* perform SVD decomposition */
        dgesvd("A",
               "A",
               &m,
               &n,
               data_matrix,
               &m,
               singular_values.data(),
               left_singular_vectors,
               &m,
               right_singular_vectors,
               &n,
               work,
               &lwork,
               &info);

        /* cleanup */
        mxFree(work);
        mxDestroyArray(in);

        /* check if call was successful */
        if(info < 0)
        {
            mexErrMsgTxt("Illegal values in arguments.");
        }
        else if(info > 0)
        {
            mexErrMsgTxt("Failed to converge.");
        }
    }

private:
    MxSingularValueDecomposition(const trrom::MxSingularValueDecomposition &);
    trrom::MxSingularValueDecomposition & operator=(const trrom::MxSingularValueDecomposition &);
};

}

#endif /* TRROM_MATLAB_INCLUDE_TRROM_MXSPECTRALDECOMPOSITION_HPP_ */
