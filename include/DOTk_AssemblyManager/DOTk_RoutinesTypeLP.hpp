/*
 * DOTk_RoutinesTypeLP.hpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ROUTINESTYPELP_HPP_
#define DOTK_ROUTINESTYPELP_HPP_

#include "DOTk_AssemblyManager.hpp"

namespace dotk
{

template<class Type>
class vector;
template<class Type>
class DOTk_ObjectiveFunction;
template<class Type>
class DOTk_InequalityConstraint;

class DOTk_RoutinesTypeLP : public dotk::DOTk_AssemblyManager
{
public:
    DOTk_RoutinesTypeLP(const std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                        const std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > & inequality_);
    virtual ~DOTk_RoutinesTypeLP();

    virtual Real objective(const std::tr1::shared_ptr<dotk::vector<Real> > & control_);
    virtual void gradient(const std::tr1::shared_ptr<dotk::vector<Real> > & control_,
                          const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_);
    virtual Real inequalityBound(const size_t index_);
    virtual Real inequalityValue(const size_t index_, const std::tr1::shared_ptr<dotk::vector<Real> > & control_);
    virtual void inequalityGradient(const size_t index_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & control_,
                                    const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_);

private:
    std::tr1::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > m_ObjectiveFunction;
    std::vector<std::tr1::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > > m_InequalityConstraint;

private:
    DOTk_RoutinesTypeLP(const dotk::DOTk_RoutinesTypeLP &);
    dotk::DOTk_RoutinesTypeLP & operator=(const dotk::DOTk_RoutinesTypeLP &);
};

}

#endif /* DOTK_ROUTINESTYPELP_HPP_ */
