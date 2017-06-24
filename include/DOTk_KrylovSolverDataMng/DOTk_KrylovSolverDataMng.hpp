/*
 * DOTk_KrylovSolverDataMng.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_KRYLOVSOLVERDATAMNG_HPP_
#define DOTK_KRYLOVSOLVERDATAMNG_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_LeftPreconditioner;
class DOTk_RightPreconditioner;
class DOTk_OrthogonalProjection;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_KrylovSolverDataMng
{
public:
    DOTk_KrylovSolverDataMng(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                             const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator);
    virtual ~DOTk_KrylovSolverDataMng();

    void setSolverType(dotk::types::krylov_solver_t aType);
    dotk::types::krylov_solver_t getSolverType() const;

    void setMaxNumSolverItr(size_t aMaxNumIterations);
    size_t getMaxNumSolverItr() const;
    const std::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;

    const std::shared_ptr<dotk::Vector<Real> > & getSolution() const;
    const std::shared_ptr<dotk::Vector<Real> > & getFirstSolution() const;
    const std::shared_ptr<dotk::Vector<Real> > & getPreviousSolution() const;

    void setResidual(dotk::types::variable_t aType, const std::shared_ptr<dotk::Vector<Real> > & aVector);
    void setResidual(const std::shared_ptr<dotk::Vector<Real> > & aVector);
    const std::shared_ptr<dotk::Vector<Real> > & getResidual() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getResidual(size_t aIndex) const;

    const std::shared_ptr<dotk::Vector<Real> > & getMatrixTimesVector() const;

    virtual const std::shared_ptr<dotk::Vector<Real> > & getLeftPrecTimesVector() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getLeftPrecTimesVector(size_t aIndex) const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getRightPrecTimesVector() const;

    virtual void setProjection(const std::shared_ptr<dotk::DOTk_OrthogonalProjection> & aProjection);
    virtual const std::shared_ptr<dotk::DOTk_OrthogonalProjection> & getProjection() const;
    virtual void setLeftPrec(const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner);
    virtual const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    virtual void setRightPrec(const std::shared_ptr<dotk::DOTk_RightPreconditioner> & aPreconditioner);
    virtual const std::shared_ptr<dotk::DOTk_RightPreconditioner> & getRightPrec() const;

private:
    size_t m_MaxNumSolverItr;
    dotk::types::krylov_solver_t m_SolverType;

    std::shared_ptr<dotk::Vector<Real> > m_Solution;
    std::shared_ptr<dotk::Vector<Real> > m_Residual;
    std::shared_ptr<dotk::Vector<Real> > m_FirstSolution;
    std::shared_ptr<dotk::Vector<Real> > m_PreviousSolution;
    std::shared_ptr<dotk::Vector<Real> > m_MatrixTimesVector;

    std::shared_ptr<dotk::DOTk_LinearOperator> m_LinearOperator;

private:
    void initialize(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);

private:
    DOTk_KrylovSolverDataMng(const dotk::DOTk_KrylovSolverDataMng &);
    dotk::DOTk_KrylovSolverDataMng & operator=(const dotk::DOTk_KrylovSolverDataMng &);
};

}

#endif /* DOTK_KRYLOVSOLVERDATAMNG_HPP_ */
