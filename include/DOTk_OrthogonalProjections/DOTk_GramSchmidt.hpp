/*
 * DOTk_GramSchmidt.hpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_GRAMSCHMIDT_HPP_
#define DOTK_GRAMSCHMIDT_HPP_

#include <vector>
#include "DOTk_OrthogonalProjection.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;

template<typename Type>
class vector;

class DOTk_GramSchmidt: public dotk::DOTk_OrthogonalProjection
{
public:
    DOTk_GramSchmidt(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t krylov_subspace_dim_);
    virtual ~DOTk_GramSchmidt();

    void gramSchmidt(size_t ortho_vector_index_, const std::tr1::shared_ptr<dotk::vector<Real> > & kernel_vector_);

    virtual void clear();
    virtual void setLinearOperatorTimesOrthoVector(size_t index_, const std::tr1::shared_ptr<dotk::vector<Real> > & vec_);
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getLinearOperatorTimesOrthoVector(size_t index_) const;
    virtual void setOrthogonalVector(size_t index_, const std::tr1::shared_ptr<dotk::vector<Real> > & vec_);
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getOrthogonalVector(size_t index_) const;

    virtual void apply(const dotk::DOTk_KrylovSolver * const solver_, const std::tr1::shared_ptr<dotk::vector<Real> > & kernel_vector_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > m_OrthogonalBasis;
    std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > m_LinearOperatorTimesOrthoVector;

private:
    DOTk_GramSchmidt(const dotk::DOTk_GramSchmidt &);
    dotk::DOTk_GramSchmidt & operator=(const dotk::DOTk_GramSchmidt &);
};

}

#endif /* DOTK_GRAMSCHMIDT_HPP_ */
