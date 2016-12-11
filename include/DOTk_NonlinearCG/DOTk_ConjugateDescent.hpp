/*
 * DOTk_ConjugateDescent.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_CONJUGATEDESCENT_HPP_
#define DOTK_CONJUGATEDESCENT_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_ConjugateDescent : public dotk::DOTk_DescentDirection
{
public:
    DOTk_ConjugateDescent();
    virtual ~DOTk_ConjugateDescent();

    Real computeScaleFactor(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_);
    void getDirection(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                      const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                      const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_);

    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    DOTk_ConjugateDescent(const dotk::DOTk_ConjugateDescent &);
    dotk::DOTk_ConjugateDescent & operator=(const dotk::DOTk_ConjugateDescent &);
};

}

#endif /* DOTK_CONJUGATEDESCENT_HPP_ */
