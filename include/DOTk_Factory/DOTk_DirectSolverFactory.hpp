/*
 * DOTk_DirectSolverFactory.hpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DIRECTSOLVERFACTORY_HPP_
#define DOTK_DIRECTSOLVERFACTORY_HPP_

namespace dotk
{

class DOTk_DirectSolver;

class DOTk_DirectSolverFactory
{
public:
    DOTk_DirectSolverFactory();
    explicit DOTk_DirectSolverFactory(dotk::types::direct_solver_t type_);
    ~DOTk_DirectSolverFactory();

    void setErrorMsg(const std::string & msg_);
    std::string getWarningMsg() const;
    void setFactoryType(dotk::types::direct_solver_t type_);
    dotk::types::direct_solver_t getFactoryType() const;

    void buildLowerTriangularDirectSolver(std::tr1::shared_ptr<dotk::DOTk_DirectSolver> & direct_solver_);
    void buildUpperTriangularDirectSolver(std::tr1::shared_ptr<dotk::DOTk_DirectSolver> & direct_solver_);
    void build(std::tr1::shared_ptr<dotk::DOTk_DirectSolver> & direct_solver_);

private:
    std::string mErrorMsg;
    dotk::types::direct_solver_t mFactoryType;

private:
    DOTk_DirectSolverFactory(const dotk::DOTk_DirectSolverFactory &);
    dotk::DOTk_DirectSolverFactory & operator=(const dotk::DOTk_DirectSolverFactory &);
};

}

#endif /* DOTK_DIRECTSOLVERFACTORY_HPP_ */
