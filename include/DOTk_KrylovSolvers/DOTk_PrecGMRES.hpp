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

template<class Type>
class vector;

class DOTk_PrecGMRES : public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_PrecGMRES(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_);
    DOTk_PrecGMRES(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                   const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                   size_t max_num_itr_);
    virtual ~DOTk_PrecGMRES();

    void initialize
    (const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
     const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
     const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_);
    void gmres(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
               const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
               const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_);

    virtual void setMaxNumKrylovSolverItr(size_t itr_);
    virtual const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getDescentDirection();
    virtual void solve(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                       const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                       const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_prob_mng_);

private:
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_DataMng;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ProjectionOperatorTimesVec;

private:
    void allocate(const std::tr1::shared_ptr<dotk::vector<Real> > vec_);

private:
    DOTk_PrecGMRES(const dotk::DOTk_PrecGMRES &);
    dotk::DOTk_PrecGMRES & operator=(const dotk::DOTk_PrecGMRES &);
};

}

#endif /* DOTK_PRECGMRES_HPP_ */
