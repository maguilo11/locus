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

template<class ScalarType>
class Vector;

class DOTk_FeasibleDirection: public dotk::DOTk_BoundConstraint
{
public:
    explicit DOTk_FeasibleDirection(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_FeasibleDirection();

    void getDirection(const std::shared_ptr<dotk::Vector<Real> > & primal_,
                      const std::shared_ptr<dotk::Vector<Real> > & feasible_dir_);

    virtual void constraint(const std::shared_ptr<dotk::DOTk_LineSearch> & step_,
                            const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    void initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    std::shared_ptr<dotk::Vector<Real> > m_LowerBounds;
    std::shared_ptr<dotk::Vector<Real> > m_UpperBounds;
    std::shared_ptr<dotk::Vector<Real> > m_TrialPrimal;

private:
    DOTk_FeasibleDirection(const dotk::DOTk_FeasibleDirection &);
    dotk::DOTk_FeasibleDirection operator=(const dotk::DOTk_FeasibleDirection &);
};

}

#endif /* DOTK_FEASIBLEDIRECTION_HPP_ */
