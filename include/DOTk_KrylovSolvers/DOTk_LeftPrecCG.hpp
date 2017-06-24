/*
 * DOTk_LeftPrecCG.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECCG_HPP_
#define DOTK_LEFTPRECCG_HPP_

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

class DOTk_LeftPrecCG : public dotk::DOTk_KrylovSolver
{
public:
    explicit DOTk_LeftPrecCG(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aSolverDataMng);
    DOTk_LeftPrecCG(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                    const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator);
    virtual ~DOTk_LeftPrecCG();

    void initialize(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                    const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                    const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);
    void pcg(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
             const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
             const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);

    virtual void setMaxNumKrylovSolverItr(size_t aMaxNumIterations);
    virtual const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & getDataMng() const;
    virtual const std::shared_ptr<dotk::DOTk_LinearOperator> & getLinearOperator() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getDescentDirection();
    virtual void solve(const std::shared_ptr<dotk::Vector<Real> > & aRhsVector,
                       const std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> & aCriterion,
                       const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);

private:
    std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> m_SolverDataMng;
    std::shared_ptr<dotk::Vector<Real> > m_ConjugateDirection;

private:
    DOTk_LeftPrecCG(const dotk::DOTk_LeftPrecCG &);
    dotk::DOTk_LeftPrecCG & operator=(const dotk::DOTk_LeftPrecCG &);
};

}

#endif /* DOTK_LEFTPRECCG_HPP_ */
