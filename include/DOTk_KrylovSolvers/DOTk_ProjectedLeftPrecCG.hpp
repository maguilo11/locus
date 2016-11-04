/*
 * DOTk_ProjectedLeftPrecCG.hpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PROJECTEDLEFTPRECCG_HPP_
#define DOTK_PROJECTEDLEFTPRECCG_HPP_

#include "DOTk_KrylovSolver.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_LinearOperator;
class DOTk_KrylovSolverDataMng;
class DOTk_OptimizationDataMng;
class DOTk_KrylovSolverStoppingCriterion;

template<class Type>
class vector;

class DOTk_ProjectedLeftPrecCG: public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_ProjectedLeftPrecCG(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_);
    DOTk_ProjectedLeftPrecCG(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                             const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & operator_,
                             size_t max_num_itr_ = 200);
    virtual ~DOTk_ProjectedLeftPrecCG();

    Real getOrthogonalityTolerance() const;
    void setOrthogonalityTolerance(Real tolerance_);
    Real getInexactnessTolerance() const;
    void setInexactnessTolerance(Real tolerance_);
    Real getOrthogonalityMeasure(size_t row_, size_t column_) const;

    virtual void setMaxNumKrylovSolverItr(size_t itr_);
    virtual const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > &
    getDescentDirection();

    bool checkOrthogonalityMeasure();
    void initialize(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                    const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    void ppcg(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
              const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
              const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    virtual void solve(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                       const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                       const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    void setProjectedConjugateDirection();
    bool checkInexactnessMeasure(Real curvature_,
                                 Real norm_old_residual_,
                                 Real old_residual_dot_projected_prec_residual_);
    void initialize();
    void setFirstSolution();
    void clear();

private:
    Real m_InexactnessTolerance;
    Real m_OrthogonalityTolerance;

    std::vector<std::vector<Real> > m_OrthogonalityMeasure;
    std::vector<Real> m_OneOverNormPreconditionerTimesResidual;
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_DataMng;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ProjectedConjugateDirection;

private:
    DOTk_ProjectedLeftPrecCG(const dotk::DOTk_ProjectedLeftPrecCG &);
    dotk::DOTk_ProjectedLeftPrecCG & operator=(const dotk::DOTk_ProjectedLeftPrecCG &);
};

}

#endif /* DOTK_PROJECTEDLEFTPRECCG_HPP_ */
