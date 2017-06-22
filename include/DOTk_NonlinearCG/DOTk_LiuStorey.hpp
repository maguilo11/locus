/*
 * DOTk_LiuStorey.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LIUSTOREY_HPP_
#define DOTK_LIUSTOREY_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_LiuStorey : public dotk::DOTk_DescentDirection
{
public:
    DOTk_LiuStorey();
    virtual ~DOTk_LiuStorey();

    Real computeScaleFactor(const std::shared_ptr<dotk::Vector<Real> > & old_grad_,
                            const std::shared_ptr<dotk::Vector<Real> > & new_grad_,
                            const std::shared_ptr<dotk::Vector<Real> > & dir_);
    void getDirection(const std::shared_ptr<dotk::Vector<Real> > & old_grad_,
                      const std::shared_ptr<dotk::Vector<Real> > & new_grad_,
                      const std::shared_ptr<dotk::Vector<Real> > & dir_);
    virtual void direction(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    DOTk_LiuStorey(const dotk::DOTk_LiuStorey &);
    dotk::DOTk_LiuStorey & operator=(const dotk::DOTk_LiuStorey &);
};

}

#endif /* DOTK_LIUSTOREY_HPP_ */
