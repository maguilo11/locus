/*
 * DOTk_DaiYuan.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DAIYUAN_HPP_
#define DOTK_DAIYUAN_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_DaiYuan : public dotk::DOTk_DescentDirection
{
public:
    DOTk_DaiYuan();
    virtual ~DOTk_DaiYuan();

    Real computeScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    void getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    DOTk_DaiYuan(const dotk::DOTk_DaiYuan &);
    dotk::DOTk_DaiYuan & operator=(const dotk::DOTk_DaiYuan &);
};

}

#endif /* DOTK_DAIYUAN_HPP_ */
