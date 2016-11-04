/*
 * DOTk_ProjectionAlongFeasibleDir.hpp
 *
 *  Created on: Sep 19, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_PROJECTIONALONGFEASIBLEDIR_HPP_
#define DOTK_PROJECTIONALONGFEASIBLEDIR_HPP_

#include "DOTk_BoundConstraint.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LineSearch;
class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_ProjectionAlongFeasibleDir: public dotk::DOTk_BoundConstraint
{
public:
    explicit DOTk_ProjectionAlongFeasibleDir(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_ProjectionAlongFeasibleDir();

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
    DOTk_ProjectionAlongFeasibleDir(const dotk::DOTk_ProjectionAlongFeasibleDir &);
    dotk::DOTk_ProjectionAlongFeasibleDir operator=(const dotk::DOTk_ProjectionAlongFeasibleDir &);
};

}

#endif /* DOTK_PROJECTIONALONGFEASIBLEDIR_HPP_ */
