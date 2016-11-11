/*
 * TRROM_LowRankSVD.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_LOWRANKSVD_HPP_
#define TRROM_LOWRANKSVD_HPP_

#include "TRROM_SpectralDecomposition.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class OrthogonalFactorization;

class LowRankSVD : public trrom::SpectralDecomposition
{
public:
    LowRankSVD();
    explicit LowRankSVD(const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_);
    LowRankSVD(const std::tr1::shared_ptr<trrom::SpectralDecomposition> & svd_,
               const std::tr1::shared_ptr<trrom::OrthogonalFactorization> & ortho_);
    virtual ~LowRankSVD();

    void formMatrixK(const trrom::Vector<double> & singular_values_,
                     const trrom::Matrix<double> & M_,
                     const trrom::Matrix<double> & R_,
                     trrom::Matrix<double> & K_);
    void formMatrixUbar(const trrom::Matrix<double> & left_singular_vectors_,
                        const trrom::Matrix<double> & Q_,
                        const trrom::Matrix<double> & C_,
                        trrom::Matrix<double> & Ubar_);
    void formMatrixVbar(int num_new_snapshots_,
                        const trrom::Matrix<double> & right_singular_vectors_,
                        const trrom::Matrix<double> & D_,
                        trrom::Matrix<double> & Vbar_);

    void solve(const std::tr1::shared_ptr<trrom::Matrix<double> > & new_snapshots_,
               std::tr1::shared_ptr<trrom::Vector<double> > & singular_values_,
               std::tr1::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
               std::tr1::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_);

private:
    std::tr1::shared_ptr<trrom::SpectralDecomposition> m_SpectralMethod;
    std::tr1::shared_ptr<trrom::OrthogonalFactorization> m_OrthoFactorization;

private:
    LowRankSVD(const trrom::LowRankSVD &);
    trrom::LowRankSVD & operator=(const trrom::LowRankSVD &);
};

}

#endif /* TRROM_LOWRANKSVD_HPP_ */
