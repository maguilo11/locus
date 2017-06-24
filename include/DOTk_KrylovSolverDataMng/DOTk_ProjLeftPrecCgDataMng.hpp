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
    DOTk_ProjLeftPrecCgDataMng(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                               const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                               size_t aMaxNumIterations );
    virtual ~DOTk_ProjLeftPrecCgDataMng();

    virtual const std::shared_ptr<dotk::Vector<Real> > & getResidual(size_t aIndex) const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getLeftPrecTimesVector(size_t aIndex) const;

    const std::shared_ptr<dotk::DOTk_OrthogonalProjection> & getProjection() const;
    void setArnoldiProjection(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);
    void setGramSchmidtProjection(const std::shared_ptr<dotk::DOTk_Primal> & primal);

    const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    void setLeftPrec(const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & aPreconditioner);

    void setAugmentedSystemPrecWithPcgSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);
    void setAugmentedSystemPrecWithGcrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);
    void setAugmentedSystemPrecWithCrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);
    void setAugmentedSystemPrecWithCgneSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);
    void setAugmentedSystemPrecWithCgnrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);
    void setAugmentedSystemPrecWithGmresSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);

private:
    void initialize(size_t aMaxNumIterations , const std::shared_ptr<dotk::Vector<Real> > aVector);

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
