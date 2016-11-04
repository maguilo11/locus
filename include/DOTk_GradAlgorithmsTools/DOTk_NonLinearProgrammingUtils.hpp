/*
 * DOTk_NonLinearProgrammingUtils.hpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_NONLINEARPROGRAMMINGUTILS_HPP_
#define DOTK_NONLINEARPROGRAMMINGUTILS_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

template<class T>
class vector;

namespace nlp
{

struct variables
{
public:
    variables(const dotk::vector<Real> & state_, const dotk::vector<Real> & control_) :
            mDual(state_.clone()),
            mState(state_.clone()),
            mControl(control_.clone())
    {
        mDual->fill(0.);
        mState->copy(state_);
        mControl->copy(control_);
    }
    variables(const dotk::vector<Real> & state_, const dotk::vector<Real> & control_, const dotk::vector<Real> & dual_) :
            mDual(dual_.clone()),
            mState(state_.clone()),
            mControl(control_.clone())
    {
        mDual->copy(dual_);
        mState->copy(state_);
        mControl->copy(control_);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > mDual;
    std::tr1::shared_ptr<dotk::vector<Real> > mState;
    std::tr1::shared_ptr<dotk::vector<Real> > mControl;

private:
    variables(const dotk::nlp::variables&);
    dotk::nlp::variables operator=(const dotk::nlp::variables&);

};

std::tr1::shared_ptr<dotk::vector<Real> > clone(dotk::nlp::variables & variables_, dotk::types::variable_t codomain_);

void resetField(const dotk::vector<Real> & data_, dotk::nlp::variables & variables_, dotk::types::derivative_t type_);

void perturbField(const Real epsilon_,
                  const dotk::vector<Real> & direction_,
                  dotk::nlp::variables & variables_,
                  dotk::types::derivative_t type_);

}

}

#endif /* DOTK_NONLINEARPROGRAMMINGUTILS_HPP_ */
