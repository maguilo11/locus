/*
 * DOTk_LineSearchQuasiNewton.hpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCHQUASINEWTON_HPP_
#define DOTK_LINESEARCHQUASINEWTON_HPP_

#include "DOTk_FirstOrderAlgorithm.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_LineSearchStepMng;
class DOTk_SecondOrderOperator;
class DOTk_FirstOrderLineSearchAlgIO;
class DOTk_LineSearchAlgorithmsDataMng;

class DOTk_LineSearchQuasiNewton : public dotk::DOTk_FirstOrderAlgorithm
{
public:
    DOTk_LineSearchQuasiNewton(const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                               const std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_);
    ~DOTk_LineSearchQuasiNewton();

    const std::shared_ptr<dotk::DOTk_SecondOrderOperator> & getInvHessianPtr() const;
    void setLbfgsSecantMethod(size_t secant_storage_ = 2);
    void setLdfpSecantMethod(size_t secant_storage_ = 2);
    void setLsr1SecantMethod(size_t secant_storage_ = 2);
    void setSr1SecantMethod();
    void setBfgsSecantMethod();
    void setBarzilaiBorweinSecantMethod();

    void printDiagnosticsAndSolutionEveryItr();
    void printDiagnosticsEveryItrAndSolutionAtTheEnd();

    void getMin();

private:
    std::shared_ptr<dotk::Vector<Real> > m_InvHessianTimesVector;

    std::shared_ptr<dotk::DOTk_FirstOrderLineSearchAlgIO> m_IO;
    std::shared_ptr<dotk::DOTk_LineSearchStepMng> m_LineSearch;
    std::shared_ptr<dotk::DOTk_SecondOrderOperator> m_InvHessian;
    std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> m_DataMng;

private:
    void initialize();
    void checkInvHessianPtr();

private:
    DOTk_LineSearchQuasiNewton(const dotk::DOTk_LineSearchQuasiNewton &);
    dotk::DOTk_LineSearchQuasiNewton & operator=(const dotk::DOTk_LineSearchQuasiNewton &);
};

}

#endif /* DOTK_LINESEARCHQUASINEWTON_HPP_ */
