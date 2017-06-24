/*
 * DOTk_DirectSolverFactory.hpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DIRECTSOLVERFACTORY_HPP_
#define DOTK_DIRECTSOLVERFACTORY_HPP_

#include <string>
#include <memory>

namespace dotk
{

class DOTk_DirectSolver;

class DOTk_DirectSolverFactory
{
public:
    DOTk_DirectSolverFactory();
    explicit DOTk_DirectSolverFactory(dotk::types::direct_solver_t aType);
    ~DOTk_DirectSolverFactory();

    void setErrorMsg(const std::string & aMsg);
    std::string getWarningMsg() const;
    void setFactoryType(dotk::types::direct_solver_t aType);
    dotk::types::direct_solver_t getFactoryType() const;

    void buildLowerTriangularDirectSolver(std::shared_ptr<dotk::DOTk_DirectSolver> & aDirectSolver);
    void buildUpperTriangularDirectSolver(std::shared_ptr<dotk::DOTk_DirectSolver> & aDirectSolver);
    void build(std::shared_ptr<dotk::DOTk_DirectSolver> & aDirectSolver);

private:
    std::string mErrorMsg;
    dotk::types::direct_solver_t mFactoryType;

private:
    DOTk_DirectSolverFactory(const dotk::DOTk_DirectSolverFactory &);
    dotk::DOTk_DirectSolverFactory & operator=(const dotk::DOTk_DirectSolverFactory &);
};

}

#endif /* DOTK_DIRECTSOLVERFACTORY_HPP_ */
