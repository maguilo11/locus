/*
 * DOTk_AugmentedSystemLeftPrec.hpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_AUGMENTEDSYSTEMLEFTPREC_HPP_
#define DOTK_AUGMENTEDSYSTEMLEFTPREC_HPP_

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_AugmentedSystem;
class DOTk_LeftPreconditioner;
class DOTk_OptimizationDataMng;
class DOTk_TangentialSubProblemCriterion;

template<typename ScalarType>
class Vector;

class DOTk_AugmentedSystemLeftPrec: public dotk::DOTk_LeftPreconditioner
{
public:
    explicit DOTk_AugmentedSystemLeftPrec(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    virtual ~DOTk_AugmentedSystemLeftPrec();

    virtual void setParameter(dotk::types::stopping_criterion_param_t type_, Real parameter_);
    virtual Real getParameter(dotk::types::stopping_criterion_param_t type_) const;
    virtual void setLeftPrecCgSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     size_t max_num_itr_ = 200);
    virtual void setLeftPrecCrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                     size_t max_num_itr_ = 200);
    virtual void setLeftPrecGcrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                      size_t max_num_itr_ = 200);
    virtual void setLeftPrecCgneSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                       size_t max_num_itr_ = 200);
    virtual void setLeftPrecCgnrSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                       size_t max_num_itr_ = 200);
    virtual void setPrecGmresSolver(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                    size_t max_num_itr_ = 200);

    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & output_);

private:
    void initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    std::tr1::shared_ptr<dotk::DOTk_KrylovSolver> m_Solver;
    std::tr1::shared_ptr<dotk::DOTk_AugmentedSystem> m_AugmentedSystem;
    std::tr1::shared_ptr<dotk::DOTk_TangentialSubProblemCriterion> m_Criterion;

    std::tr1::shared_ptr<dotk::Vector<Real> > m_RhsVector;

private:
    DOTk_AugmentedSystemLeftPrec(const dotk::DOTk_AugmentedSystemLeftPrec &);
    dotk::DOTk_AugmentedSystemLeftPrec & operator=(const dotk::DOTk_AugmentedSystemLeftPrec &);
};

}

#endif /* DOTK_AUGMENTEDSYSTEMLEFTPREC_HPP_ */
