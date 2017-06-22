/*
 * DOTk_DataMngNonlinearCG.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DATAMNGNONLINEARCG_HPP_
#define DOTK_DATAMNGNONLINEARCG_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_DataMngNonlinearCG
{
public:
    explicit DOTk_DataMngNonlinearCG(const std::shared_ptr<dotk::Vector<Real> > & dual_);
    virtual ~DOTk_DataMngNonlinearCG();

    void reset();
    void storeCurrentState();

public:
    Real m_NewObjectiveFunctionValue;
    Real m_OldObjectiveFunctionValue;

    std::shared_ptr<dotk::Vector<Real> > m_NewDual;
    std::shared_ptr<dotk::Vector<Real> > m_OldDual;
    std::shared_ptr<dotk::Vector<Real> > m_NewGradient;
    std::shared_ptr<dotk::Vector<Real> > m_OldGradient;
    std::shared_ptr<dotk::Vector<Real> > m_NewTrialStep;
    std::shared_ptr<dotk::Vector<Real> > m_OldTrialStep;
    std::shared_ptr<dotk::Vector<Real> > m_NewSteepestDescent;
    std::shared_ptr<dotk::Vector<Real> > m_OldSteepestDescent;

private:
    DOTk_DataMngNonlinearCG(const dotk::DOTk_DataMngNonlinearCG &);
    dotk::DOTk_DataMngNonlinearCG & operator=(const dotk::DOTk_DataMngNonlinearCG & rhs_);
};

}

#endif /* DOTK_DATAMNGNONLINEARCG_HPP_ */
