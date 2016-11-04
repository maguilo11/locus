/*
 * DOTk_LeftPrecCGNE.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECCGNE_HPP_
#define DOTK_LEFTPRECCGNE_HPP_

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

class DOTk_LeftPrecCGNE : public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_LeftPrecCGNE(const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_);
    DOTk_LeftPrecCGNE(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                      const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_);
    virtual ~DOTk_LeftPrecCGNE();

    void initialize(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                    const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                    const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);
    void cgne(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
              const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
              const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);

    virtual void setMaxNumKrylovSolverItr(size_t itr_);
    virtual const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getDescentDirection();
    virtual void solve(const std::tr1::shared_ptr<dotk::vector<Real> > & rhs_vec_,
                       const std::tr1::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & criterion_,
                       const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & opt_mng_);

private:
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_DataMng;
    std::tr1::shared_ptr<dotk::vector<Real> > m_AuxiliaryVector;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ConjugateDirection;
    std::tr1::shared_ptr<dotk::vector<Real> > m_ConjugateDirectionNormalEq;

private:
    void initialize(const std::tr1::shared_ptr<dotk::vector<Real> > vec_);

private:
    DOTk_LeftPrecCGNE(const dotk::DOTk_LeftPrecCGNE &);
    dotk::DOTk_LeftPrecCGNE & operator=(const dotk::DOTk_LeftPrecCGNE &);
};

}

#endif /* DOTK_LEFTPRECCGNE_HPP_ */
