/*
 * DOTk_NonLinearProgrammingUtils.hpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NONLINEARPROGRAMMINGUTILS_HPP_
#define DOTK_NONLINEARPROGRAMMINGUTILS_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

namespace nlp
{

struct variables
{
public:
    variables(const dotk::Vector<Real> & state_, const dotk::Vector<Real> & control_) :
            mDual(state_.clone()),
            mState(state_.clone()),
            mControl(control_.clone())
    {
        mDual->fill(0.);
        mState->update(1., state_, 0.);
        mControl->update(1., control_, 0.);
    }
    variables(const dotk::Vector<Real> & state_, const dotk::Vector<Real> & control_, const dotk::Vector<Real> & dual_) :
            mDual(dual_.clone()),
            mState(state_.clone()),
            mControl(control_.clone())
    {
        mDual->update(1., dual_, 0.);
        mState->update(1., state_, 0.);
        mControl->update(1., control_, 0.);
    }

    std::shared_ptr<dotk::Vector<Real> > mDual;
    std::shared_ptr<dotk::Vector<Real> > mState;
    std::shared_ptr<dotk::Vector<Real> > mControl;

private:
    variables(const dotk::nlp::variables&);
    dotk::nlp::variables operator=(const dotk::nlp::variables&);

};

std::shared_ptr<dotk::Vector<Real> > clone(dotk::nlp::variables & variables_, dotk::types::variable_t codomain_);

void resetField(const dotk::Vector<Real> & data_, dotk::nlp::variables & variables_, dotk::types::derivative_t type_);

void perturbField(const Real epsilon_,
                  const dotk::Vector<Real> & direction_,
                  dotk::nlp::variables & variables_,
                  dotk::types::derivative_t type_);

}

}

#endif /* DOTK_NONLINEARPROGRAMMINGUTILS_HPP_ */
