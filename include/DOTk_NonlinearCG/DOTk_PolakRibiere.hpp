/*
 * DOTk_PolakRibiere.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_POLAKRIBIERE_HPP_
#define DOTK_POLAKRIBIERE_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_PolakRibiere : public dotk::DOTk_DescentDirection
{
public:
    DOTk_PolakRibiere();
    virtual ~DOTk_PolakRibiere();

    Real computeScaleFactor(const std::shared_ptr<dotk::Vector<Real> > & old_grad_,
                            const std::shared_ptr<dotk::Vector<Real> > & new_grad_);
    void getDirection(const std::shared_ptr<dotk::Vector<Real> > & old_grad_,
                      const std::shared_ptr<dotk::Vector<Real> > & new_grad_,
                      const std::shared_ptr<dotk::Vector<Real> > & dir_);
    virtual void direction(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    DOTk_PolakRibiere(const dotk::DOTk_PolakRibiere &);
    dotk::DOTk_PolakRibiere & operator=(const dotk::DOTk_PolakRibiere &);
};

}

#endif /* DOTK_POLAKRIBIERE_HPP_ */
