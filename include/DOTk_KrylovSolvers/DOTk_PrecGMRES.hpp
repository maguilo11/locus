/*
 * DOTk_PrecGMRES.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PRECGMRES_HPP_
#define DOTK_PRECGMRES_HPP_

#include "DOTk_KrylovSolver.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_KrylovSolverDataMng;
class DOTk_OptimizationDataMng;
class DOTk_KrylovSolverStoppingCriterion;

template<typename ScalarType>
class Vector;

class DOTk_PrecGMRES : public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_PrecGMRES(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_);
    DOTk_PrecGMRES(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                   const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                   size_t max_num_itr_);
    virtual ~DOTk_PrecGMRES();

    void initialize
    (const std::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
     const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
     const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_);
    void gmres(const std::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
               const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
               const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_);

    virtual void setMaxNumKrylovSolverItr(size_t itr_);
    virtual const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getDescentDirection();
    virtual void solve(const std::shared_ptr<dotk::Vector<Real> > & rhs_vec_,
                       const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                       const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_);

private:
    std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_DataMng;
    std::shared_ptr<dotk::Vector<Real> > m_ProjectionOperatorTimesVec;

private:
    void allocate(const std::shared_ptr<dotk::Vector<Real> > vec_);

private:
    DOTk_PrecGMRES(const dotk::DOTk_PrecGMRES &);
    dotk::DOTk_PrecGMRES & operator=(const dotk::DOTk_PrecGMRES &);
};

}

#endif /* DOTK_PRECGMRES_HPP_ */
