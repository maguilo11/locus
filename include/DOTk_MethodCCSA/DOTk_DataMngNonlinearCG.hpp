/*
 * DOTk_DataMngNonlinearCG.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DATAMNGNONLINEARCG_HPP_
#define DOTK_DATAMNGNONLINEARCG_HPP_

#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_DataMngNonlinearCG
{
public:
    explicit DOTk_DataMngNonlinearCG(const std::tr1::shared_ptr<dotk::Vector<Real> > & dual_);
    virtual ~DOTk_DataMngNonlinearCG();

    void reset();
    void storeCurrentState();

public:
    Real m_NewObjectiveFunctionValue;
    Real m_OldObjectiveFunctionValue;

    std::tr1::shared_ptr<dotk::Vector<Real> > m_NewDual;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OldDual;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_NewGradient;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OldGradient;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_NewTrialStep;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OldTrialStep;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_NewSteepestDescent;
    std::tr1::shared_ptr<dotk::Vector<Real> > m_OldSteepestDescent;

private:
    DOTk_DataMngNonlinearCG(const dotk::DOTk_DataMngNonlinearCG &);
    dotk::DOTk_DataMngNonlinearCG & operator=(const dotk::DOTk_DataMngNonlinearCG & rhs_);
};

}

#endif /* DOTK_DATAMNGNONLINEARCG_HPP_ */
