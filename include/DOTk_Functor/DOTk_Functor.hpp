/*
 * DOTk_Functor.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_FUNCTOR_HPP_
#define DOTK_FUNCTOR_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

template<class Type>
class vector;

class DOTk_Functor
{
public:
    explicit DOTk_Functor(dotk::types::functor_t type_);
    virtual ~DOTk_Functor();

    dotk::types::functor_t getFunctorType() const;
    virtual void operator()(const dotk::vector<Real> & control_, dotk::vector<Real> & output_);
    virtual void operator()(const dotk::vector<Real> & state_,
                            const dotk::vector<Real> & control_,
                            const dotk::vector<Real> & dual_,
                            dotk::vector<Real> & output_);

private:
    dotk::types::functor_t m_FunctorType;

private:
    DOTk_Functor(const dotk::DOTk_Functor&);
    dotk::DOTk_Functor operator=(const dotk::DOTk_Functor&);
};

}

#endif /* DOTK_FUNCTOR_HPP_ */
