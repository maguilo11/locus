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

template<class Type>
class vector;

class DOTk_LiuStorey : public dotk::DOTk_DescentDirection
{
public:
    DOTk_LiuStorey();
    virtual ~DOTk_LiuStorey();

    Real computeScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    void getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    DOTk_LiuStorey(const dotk::DOTk_LiuStorey &);
    dotk::DOTk_LiuStorey & operator=(const dotk::DOTk_LiuStorey &);
};

}

#endif /* DOTK_LIUSTOREY_HPP_ */
