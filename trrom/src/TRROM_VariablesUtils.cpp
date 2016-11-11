/*
 * TRROM_VariablesUtils.cpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <fstream>
#include <sstream>

#include "TRROM_Vector.hpp"
#include "TRROM_VariablesUtils.hpp"

namespace trrom
{

void printDual(const std::tr1::shared_ptr<trrom::Vector<double> > & dual_)
{
    std::ofstream file("dual_solution.out", std::ios::out | std::ios::trunc);
    for(int i = 0; i < dual_->size(); ++i)
    {
        file << (*dual_)[i] << "\n";
    }
    file.close();
}

void printState(const std::tr1::shared_ptr<trrom::Vector<double> > & state_)
{
    std::ofstream file("state_solution.out", std::ios::out | std::ios::trunc);
    for(int i = 0; i < state_->size(); ++i)
    {
        file << (*state_)[i] << "\n";
    }
    file.close();
}

void printControl(const std::tr1::shared_ptr<trrom::Vector<double> > & control_)
{
    std::ofstream file("control_solution.out", std::ios::out | std::ios::trunc);
    for(int i = 0; i < control_->size(); ++i)
    {
        file << (*control_)[i] << "\n";
    }
    file.close();
}

void printSolution(const std::tr1::shared_ptr<trrom::Vector<double> > & solution_)
{
    std::ofstream file("solution.out", std::ios::out | std::ios::trunc);
    for(int i = 0; i < solution_->size(); ++i)
    {
        file << (*solution_)[i] << "\n";
    }
    file.close();
}

}
