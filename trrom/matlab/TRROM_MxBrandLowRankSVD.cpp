/*
 * TRROM_MxBrandLowRankSVD.cpp
 *
 *  Created on: Dec 7, 2016
 *      Author: maguilo
 */

#include "TRROM_Vector.hpp"
#include "TRROM_Matrix.hpp"
#include "TRROM_BrandLowRankSVD.hpp"
#include "TRROM_MxBrandLowRankSVD.hpp"
#include "TRROM_MxBrandMatrixFactory.hpp"
#include "TRROM_MxOrthogonalDecomposition.hpp"
#include "TRROM_MxSingularValueDecomposition.hpp"

namespace trrom
{

MxBrandLowRankSVD::MxBrandLowRankSVD() :
        m_Algorithm(),
        m_MatrixFactory(new trrom::MxBrandMatrixFactory),
        m_SpectralMethod(new trrom::MxSingularValueDecomposition),
        m_OrthoFactorization(new trrom::MxOrthogonalDecomposition)
{
    m_Algorithm.reset(new trrom::BrandLowRankSVD(m_MatrixFactory, m_SpectralMethod, m_OrthoFactorization));
}

MxBrandLowRankSVD::~MxBrandLowRankSVD()
{
}

void MxBrandLowRankSVD::solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & data_set_,
                              std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
                              std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
                              std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_)
{
    m_Algorithm->solve(data_set_, singular_values_, left_singular_vectors_, right_singular_vectors_);
}

}
