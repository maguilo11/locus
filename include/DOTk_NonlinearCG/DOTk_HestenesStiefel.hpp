/*
 * DOTk_HestenesStiefel.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_HESTENESSTIEFEL_HPP_
#define DOTK_HESTENESSTIEFEL_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_HestenesStiefel : public dotk::DOTk_DescentDirection
{
public:
    DOTk_HestenesStiefel();
    virtual ~DOTk_HestenesStiefel();

    Real computeScaleFactor(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_);
    void getDirection(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                      const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                      const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_);
    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    DOTk_HestenesStiefel(const dotk::DOTk_HestenesStiefel &);
    dotk::DOTk_HestenesStiefel & operator=(const dotk::DOTk_HestenesStiefel &);
};

}

#endif /* DOTK_HESTENESSTIEFEL_HPP_ */
