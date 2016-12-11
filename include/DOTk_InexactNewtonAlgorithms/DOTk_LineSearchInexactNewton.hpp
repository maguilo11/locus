/*
 * DOTk_LineSearchInexactNewton.hpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCHINEXACTNEWTON_HPP_
#define DOTK_LINESEARCHINEXACTNEWTON_HPP_

#include "DOTk_InexactNewtonAlgorithms.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_LinearOperator;
class DOTk_LineSearchStepMng;
class DOTk_LineSearchInexactNewtonIO;
class DOTk_LineSearchAlgorithmsDataMng;

template<typename ScalarType>
class Vector;

class DOTk_LineSearchInexactNewton : public dotk::DOTk_InexactNewtonAlgorithms
{
public:
    DOTk_LineSearchInexactNewton(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
                                 const std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                                 const std::tr1::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_);
    virtual ~DOTk_LineSearchInexactNewton();

    virtual void setNumItrDone(size_t itr_);
    void setMaxNumKrylovSolverItr(size_t itr_);

    void setPrecGmresKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t max_num_itr_ = 200);
    void setLeftPrecCgKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t max_num_itr_ = 200);
    void setLeftPrecCrKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t max_num_itr_ = 200);
    void setLeftPrecGcrKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_, size_t max_num_itr_ = 200);
    void setLeftPrecCgneKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     size_t max_num_itr_ = 200);
    void setLeftPrecCgnrKrylovSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     size_t max_num_itr_ = 200);

    void printDiagnosticsAndSolutionEveryItr();
    void printDiagnosticsEveryItrAndSolutionAtTheEnd();

    void getMin();

private:
    void checkAlgorithmInputs();

private:
    std::tr1::shared_ptr<dotk::Vector<Real> > m_SolverRhsVector;

    std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> m_KrylovSolver;
    std::tr1::shared_ptr<dotk::DOTk_LineSearchInexactNewtonIO> m_IO;
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> m_LineSearch;
    std::tr1::shared_ptr<dotk::DOTk_LinearOperator> m_LinearOperator;
    std::tr1::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> m_DataMng;

private:
    DOTk_LineSearchInexactNewton(const dotk::DOTk_LineSearchInexactNewton&);
    dotk::DOTk_LineSearchInexactNewton operator=(const dotk::DOTk_LineSearchInexactNewton&);
};

}

#endif /* DOTK_LINESEARCHINEXACTNEWTON_HPP_ */
