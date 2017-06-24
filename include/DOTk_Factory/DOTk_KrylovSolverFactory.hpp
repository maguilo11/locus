/*
 * DOTk_KrylovSolverFactory.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_KRYLOVSOLVERFACTORY_HPP_
#define DOTK_KRYLOVSOLVERFACTORY_HPP_

#include <string>
#include <memory>

namespace dotk
{

class DOTk_Primal;
class DOTk_KrylovSolver;
class DOTk_LinearOperator;

class DOTk_KrylovSolverFactory
{
public:
    DOTk_KrylovSolverFactory();
    explicit DOTk_KrylovSolverFactory(dotk::types::krylov_solver_t aType);
    ~DOTk_KrylovSolverFactory();

    void setWarningMsg(const std::string & aMsg);
    std::string getWarningMsg() const;
    dotk::types::krylov_solver_t getFactoryType() const;
    void setFactoryType(dotk::types::krylov_solver_t aType);

    void buildPrecGmresSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                              const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                              size_t aMaxNumIterations,
                              std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput);
    void buildLeftPrecCgSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                               const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                               std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput);
    void buildLeftPrecCrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                               const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                               std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput);
    void buildLeftPrecGcrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                size_t aMaxNumIterations,
                                std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput);
    void buildLeftPrecCgneSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                 const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                 std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput);
    void buildLeftPrecCgnrSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                 const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                 std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput);
    void buildProjLeftPrecCgSolver(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                                   const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                                   size_t aMaxNumIterations,
                                   std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput);
    void build(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & aMng,
               std::shared_ptr<dotk::DOTk_KrylovSolver> & aOutput);

private:
    std::string mWarningMsg;
    dotk::types::krylov_solver_t mFactoryType;

private:
    DOTk_KrylovSolverFactory(const dotk::DOTk_KrylovSolverFactory &);
    dotk::DOTk_KrylovSolverFactory operator=(const dotk::DOTk_KrylovSolverFactory &);
};

}

#endif /* DOTK_KRYLOVSOLVERFACTORY_HPP_ */
