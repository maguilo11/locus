/*
 * DOTk_FletcherReeves.hpp
 *
 *  Created on: Sep 11, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_FLETCHERREEVES_HPP_
#define DOTK_FLETCHERREEVES_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_FletcherReeves : public dotk::DOTk_DescentDirection
{
public:
    DOTk_FletcherReeves();
    virtual ~DOTk_FletcherReeves();

    Real computeScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_);
    void getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    DOTk_FletcherReeves(const dotk::DOTk_FletcherReeves &);
    dotk::DOTk_FletcherReeves & operator=(const dotk::DOTk_FletcherReeves &);
};

}

#endif /* DOTK_FLETCHERREEVES_HPP_ */
