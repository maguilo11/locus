/*
 * DOTk_FeasibleDirection.hpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_FEASIBLEDIRECTION_HPP_
#define DOTK_FEASIBLEDIRECTION_HPP_

#include "DOTk_BoundConstraint.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LineSearch;
class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_FeasibleDirection: public dotk::DOTk_BoundConstraint
{
public:
    explicit DOTk_FeasibleDirection(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_FeasibleDirection();

    void getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & feasible_dir_);

    virtual void constraint(const std::tr1::shared_ptr<dotk::DOTk_LineSearch> & step_,
                            const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    std::tr1::shared_ptr<dotk::vector<Real> > m_LowerBounds;
    std::tr1::shared_ptr<dotk::vector<Real> > m_UpperBounds;
    std::tr1::shared_ptr<dotk::vector<Real> > m_TrialPrimal;

private:
    DOTk_FeasibleDirection(const dotk::DOTk_FeasibleDirection &);
    dotk::DOTk_FeasibleDirection operator=(const dotk::DOTk_FeasibleDirection &);
};

}

#endif /* DOTK_FEASIBLEDIRECTION_HPP_ */