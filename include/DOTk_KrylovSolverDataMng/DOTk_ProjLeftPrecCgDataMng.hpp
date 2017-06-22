/*
 * DOTk_ProjLeftPrecCgDataMng.hpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PROJLEFTPRECCGDATAMNG_HPP_
#define DOTK_PROJLEFTPRECCGDATAMNG_HPP_

#include "DOTk_KrylovSolverDataMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_LeftPreconditioner;
class DOTk_OrthogonalProjection;

template<typename ScalarType>
class Vector;

class DOTk_ProjLeftPrecCgDataMng : public dotk::DOTk_KrylovSolverDataMng
{
public:
    DOTk_ProjLeftPrecCgDataMng(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                               const std::shared_ptr<dotk::DOTk_LinearOperator> & operator_,
                               size_t max_num_itr_);
    virtual ~DOTk_ProjLeftPrecCgDataMng();

    virtual const std::shared_ptr<dotk::Vector<Real> > & getResidual(size_t index_) const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getLeftPrecTimesVector(size_t index_) const;

    const std::shared_ptr<dotk::DOTk_OrthogonalProjection> & getProjection() const;
    void setArnoldiProjection(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void setGramSchmidtProjection(const std::shared_ptr<dotk::DOTk_Primal> & primal);

    const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    void setLeftPrec(const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & prec_);

    void setAugmentedSystemPrecWithPcgSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void setAugmentedSystemPrecWithGcrSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void setAugmentedSystemPrecWithCrSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void setAugmentedSystemPrecWithCgneSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void setAugmentedSystemPrecWithCgnrSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_);
    void setAugmentedSystemPrecWithGmresSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    void initialize(size_t max_num_itr_, const std::shared_ptr<dotk::Vector<Real> > vector_);

private:
    std::shared_ptr<dotk::DOTk_OrthogonalProjection> m_ProjectionMethod;
    std::shared_ptr<dotk::DOTk_LeftPreconditioner> m_LeftPreconditioner;

    std::vector<std::shared_ptr<dotk::Vector<Real> > > m_Residual;
    std::vector<std::shared_ptr<dotk::Vector<Real> > > m_LeftPrecTimesResidual;

private:
    DOTk_ProjLeftPrecCgDataMng(const dotk::DOTk_ProjLeftPrecCgDataMng &);
    dotk::DOTk_ProjLeftPrecCgDataMng & operator=(const dotk::DOTk_ProjLeftPrecCgDataMng &);
};

}

#endif /* DOTK_PROJLEFTPRECCGDATAMNG_HPP_ */
