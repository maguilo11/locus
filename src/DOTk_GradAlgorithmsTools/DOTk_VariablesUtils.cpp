/*
 * DOTk_VariablesUtils.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <fstream>
#include <sstream>

#include "vector.hpp"
#include "DOTk_VariablesUtils.hpp"

namespace dotk
{

void printDual(const std::tr1::shared_ptr<dotk::Vector<Real> > & dual_)
{
    std::ofstream file("DOTk_dual_solution.out", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < dual_->size(); ++i)
    {
        file << (*dual_)[i] << "\n";
    }
    file.close();
}

void printState(const std::tr1::shared_ptr<dotk::Vector<Real> > & state_)
{
    std::ofstream file("DOTk_state_solution.out", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < state_->size(); ++i)
    {
        file << (*state_)[i] << "\n";
    }
    file.close();
}

void printControl(const std::tr1::shared_ptr<dotk::Vector<Real> > & control_)
{
    std::ofstream file("DOTk_control_solution.out", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < control_->size(); ++i)
    {
        file << (*control_)[i] << "\n";
    }
    file.close();
}

void printSolution(const std::tr1::shared_ptr<dotk::Vector<Real> > & solution_)
{
    std::ofstream file("DOTk_solution.out", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < solution_->size(); ++i)
    {
        file << (*solution_)[i] << "\n";
    }
    file.close();
}

}
