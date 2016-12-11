/*
 * DOTk_SteihaugTointStepMng.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STEIHAUGTOINTSTEPMNG_HPP_
#define DOTK_STEIHAUGTOINTSTEPMNG_HPP_

#include "DOTk_TrustRegionStepMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_Preconditioner;
class DOTk_SteihaugTointSolver;
class DOTk_OptimizationDataMng;
class DOTk_SteihaugTointNewtonIO;

template<typename ScalarType>
class Vector;

class DOTk_SteihaugTointStepMng : public dotk::DOTk_TrustRegionStepMng
{
public:
    DOTk_SteihaugTointStepMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_);
    DOTk_SteihaugTointStepMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                              const std::tr1::shared_ptr<dotk::DOTk_Preconditioner> & preconditioner_);
    virtual ~DOTk_SteihaugTointStepMng();

    virtual void setNumOptimizationItrDone(const size_t & itr_);
    virtual void solveSubProblem(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                 const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                 const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointNewtonIO> & io_);

private:
    void updateDataManager(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_CurrentPrimal;
    std::tr1::shared_ptr<dotk::DOTk_LinearOperator> m_LinearOperator;
    std::tr1::shared_ptr<dotk::DOTk_Preconditioner> m_Preconditioner;

private:
    DOTk_SteihaugTointStepMng(const dotk::DOTk_SteihaugTointStepMng &);
    dotk::DOTk_SteihaugTointStepMng & operator=(const dotk::DOTk_SteihaugTointStepMng & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTSTEPMNG_HPP_ */
