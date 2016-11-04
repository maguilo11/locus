/*
 * DOTk_NonlinearCG.hpp
 *
 *  Created on: Sep 18, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NONLINEARCG_HPP_
#define DOTK_NONLINEARCG_HPP_

#include "DOTk_FirstOrderAlgorithm.hpp"

namespace dotk
{

class DOTk_LinearOperator;
class DOTk_DescentDirection;
class DOTk_LineSearchStepMng;
class DOTk_FirstOrderLineSearchAlgIO;
class DOTk_LineSearchAlgorithmsDataMng;

template<class Type>
class vector;

class DOTk_NonlinearCG : public dotk::DOTk_FirstOrderAlgorithm
{
public:
    DOTk_NonlinearCG(const std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                     const std::tr1::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_);
    ~DOTk_NonlinearCG();

    Real getLineSearchStepSize() const;
    size_t getNumLineSearchItrDone() const;

    void setFletcherReevesNlcg();
    void setPolakRibiereNlcg();
    void setHestenesStiefelNlcg();
    void setConjugateDescentNlcg();
    void setHagerZhangNlcg();
    void setDaiLiaoNlcg();
    void setDaiYuanNlcg();
    void setDaiYuanHybridNlcg();
    void setPerryShannoNlcg();
    void setLiuStoreyNlcg();
    void setDanielsNlcg(const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & hessian_);

    void printDiagnosticsAndSolutionEveryItr();
    void printDiagnosticsEveryItrAndSolutionAtTheEnd();

    void getMin();

private:
    void initialize();

private:
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> m_LineSearch;
    std::tr1::shared_ptr<dotk::DOTk_FirstOrderLineSearchAlgIO> m_IO;
    std::tr1::shared_ptr<dotk::DOTk_DescentDirection> m_DescentDirection;
    std::tr1::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> m_DataMng;

private:
    DOTk_NonlinearCG(const dotk::DOTk_NonlinearCG &);
    DOTk_NonlinearCG & operator=(const dotk::DOTk_NonlinearCG &);
};

}

#endif /* DOTK_NONLINEARCG_HPP_ */
