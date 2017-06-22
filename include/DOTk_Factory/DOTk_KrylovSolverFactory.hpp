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
    explicit DOTk_KrylovSolverFactory(dotk::types::krylov_solver_t type_);
    ~DOTk_KrylovSolverFactory();

    void setWarningMsg(const std::string & msg_);
    std::string getWarningMsg() const;
    dotk::types::krylov_solver_t getFactoryType() const;
    void setFactoryType(dotk::types::krylov_solver_t type_);

    void buildPrecGmresSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                              size_t max_num_itr_,
                              std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_);
    void buildLeftPrecCgSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                               const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                               std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_);
    void buildLeftPrecCrSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                               const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                               std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_);
    void buildLeftPrecGcrSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                size_t max_num_itr_,
                                std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_);
    void buildLeftPrecCgneSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                 const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                 std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_);
    void buildLeftPrecCgnrSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                 const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                 std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_);
    void buildProjLeftPrecCgSolver(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                   const std::shared_ptr<dotk::DOTk_LinearOperator> & linear_operator_,
                                   size_t max_num_itr_,
                                   std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_);
    void build(const std::shared_ptr<dotk::DOTk_KrylovSolverDataMng> & mng_,
               std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_);

private:
    std::string mWarningMsg;
    dotk::types::krylov_solver_t mFactoryType;

private:
    DOTk_KrylovSolverFactory(const dotk::DOTk_KrylovSolverFactory &);
    dotk::DOTk_KrylovSolverFactory operator=(const dotk::DOTk_KrylovSolverFactory &);
};

}

#endif /* DOTK_KRYLOVSOLVERFACTORY_HPP_ */
