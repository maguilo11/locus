/*
 * TRROM_TeuchosSVD.hpp
 *
 *  Created on: Oct 19, 2016
 *      Author: maguilo
 */

#ifndef TRROM_TEUCHOSSVD_HPP_
#define TRROM_TEUCHOSSVD_HPP_

#include "Teuchos_LAPACK.hpp"
#include "TRROM_SpectralDecomposition.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class TeuchosSVD : public trrom::SpectralDecomposition
{
public:
    TeuchosSVD();
    virtual ~TeuchosSVD();

    void solve(const std::shared_ptr<trrom::Matrix<double> > & data_,
               std::shared_ptr<trrom::Vector<double> > & singular_values_,
               std::shared_ptr<trrom::Matrix<double> > & left_singular_vectors_,
               std::shared_ptr<trrom::Matrix<double> > & right_singular_vectors_);

private:
    Teuchos::LAPACK<int, double> m_LAPACK;

private:
    TeuchosSVD(const trrom::TeuchosSVD &);
    trrom::TeuchosSVD & operator=(const trrom::TeuchosSVD & rhs_);
};

}

#endif /* TRROM_TEUCHOSSVD_HPP_ */
