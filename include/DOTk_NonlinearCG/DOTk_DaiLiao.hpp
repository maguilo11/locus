/*
 * DOTk_DaiLiao.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DAILIAO_HPP_
#define DOTK_DAILIAO_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_DaiLiao : public dotk::DOTk_DescentDirection
{
public:
    DOTk_DaiLiao();
    virtual ~DOTk_DaiLiao();

    Real getConstant() const;
    void setConstant(Real value_);
    Real computeScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & old_primal_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & new_primal_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    void getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & old_primal_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_primal_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    Real mConstant;

private:
    DOTk_DaiLiao(const dotk::DOTk_DaiLiao &);
    dotk::DOTk_DaiLiao & operator=(const dotk::DOTk_DaiLiao &);
};

}

#endif /* DOTK_DAILIAO_HPP_ */
