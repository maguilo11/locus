/*
 * DOTk_KrylovSolverDataMng.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_KRYLOVSOLVERDATAMNG_HPP_
#define DOTK_KRYLOVSOLVERDATAMNG_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_LeftPreconditioner;
class DOTk_RightPreconditioner;
class DOTk_OrthogonalProjection;
class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_KrylovSolverDataMng
{
public:
    DOTk_KrylovSolverDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                             const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_);
    virtual ~DOTk_KrylovSolverDataMng();

    void setSolverType(dotk::types::krylov_solver_t type_);
    dotk::types::krylov_solver_t getSolverType() const;

    void setMaxNumSolverItr(size_t max_num_itr_);
    size_t getMaxNumSolverItr() const;
    const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;

    const std::tr1::shared_ptr<dotk::vector<Real> > & getSolution() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getFirstSolution() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getPreviousSolution() const;

    void setResidual(dotk::types::variable_t type_, const std::tr1::shared_ptr<dotk::vector<Real> > & vec_);
    void setResidual(const std::tr1::shared_ptr<dotk::vector<Real> > & vec_);
    const std::tr1::shared_ptr<dotk::vector<Real> > & getResidual() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getResidual(size_t index_) const;

    const std::tr1::shared_ptr<dotk::vector<Real> > & getMatrixTimesVector() const;

    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getLeftPrecTimesVector() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getLeftPrecTimesVector(size_t index_) const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getRightPrecTimesVector() const;

    virtual void setProjection(const std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & projection_);
    virtual const std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & getProjection() const;
    virtual void setLeftPrec(const std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & preconditioner_);
    virtual const std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    virtual void setRightPrec(const std::tr1::shared_ptr<dotk::DOTk_RightPreconditioner> & preconditioner_);
    virtual const std::tr1::shared_ptr<dotk::DOTk_RightPreconditioner> & getRightPrec() const;

private:
    size_t m_MaxNumSolverItr;
    dotk::types::krylov_solver_t m_SolverType;

    std::tr1::shared_ptr<dotk::vector<Real> > m_Solution;
    std::tr1::shared_ptr<dotk::vector<Real> > m_Residual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_FirstSolution;
    std::tr1::shared_ptr<dotk::vector<Real> > m_PreviousSolution;
    std::tr1::shared_ptr<dotk::vector<Real> > m_MatrixTimesVector;

    std::tr1::shared_ptr<dotk::DOTk_LinearOperator> m_LinearOperator;

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    DOTk_KrylovSolverDataMng(const dotk::DOTk_KrylovSolverDataMng &);
    dotk::DOTk_KrylovSolverDataMng & operator=(const dotk::DOTk_KrylovSolverDataMng &);
};

}

#endif /* DOTK_KRYLOVSOLVERDATAMNG_HPP_ */
