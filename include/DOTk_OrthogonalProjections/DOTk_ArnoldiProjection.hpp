/*
 * DOTk_ArnoldiProjection.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ARNOLDIPROJECTION_HPP_
#define DOTK_ARNOLDIPROJECTION_HPP_

#include <vector>
#include "DOTk_OrthogonalProjection.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_DirectSolver;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_ArnoldiProjection: public dotk::DOTk_OrthogonalProjection
{
public:
    DOTk_ArnoldiProjection(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t krylov_subspace_dim_);
    virtual ~DOTk_ArnoldiProjection();

    virtual void setInitialResidual(Real value_);
    virtual Real getInitialResidual() const;

    virtual void clear();
    virtual const std::tr1::shared_ptr<dotk::Vector<Real> > & getOrthogonalVector(size_t index_) const;
    virtual void setOrthogonalVector(size_t index_, const std::tr1::shared_ptr<dotk::Vector<Real> > & vec_);
    virtual void apply(const dotk::DOTk_KrylovSolver * const solver_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_);

    void updateHessenbergMatrix(size_t current_itr_,
                                const std::tr1::shared_ptr<dotk::Vector<Real> > & left_prec_times_vec_);
    void arnoldi(size_t ortho_vector_index_, const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_);
    void applyGivensRotationsToHessenbergMatrix(int current_itr_);

private:
    std::vector<Real> m_Sine;
    std::vector<Real> m_Cosine;
    std::vector<Real> m_ScaleFactorsStorage;
    std::vector<Real> m_NormAppxProjectedResidualStorage;
    std::tr1::shared_ptr<dotk::DOTk_DirectSolver> m_DirectSolver;
    std::tr1::shared_ptr<dotk::matrix<Real> > m_HessenbergMatrix;
    std::vector<std::tr1::shared_ptr<dotk::Vector<Real> > > m_OrthogonalBasis;

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    DOTk_ArnoldiProjection(const dotk::DOTk_ArnoldiProjection &);
    dotk::DOTk_ArnoldiProjection & operator=(const dotk::DOTk_ArnoldiProjection &);
};

}

#endif /* DOTK_ARNOLDIPROJECTION_HPP_ */
