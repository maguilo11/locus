/*
 * DOTk_Rosenbrock.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ROSENBROCK_HPP_
#define DOTK_ROSENBROCK_HPP_

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_Rosenbrock : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_Rosenbrock();
    virtual ~DOTk_Rosenbrock();

    Real value(const dotk::Vector<Real> & primal_);
    void value(const std::vector<std::shared_ptr<dotk::Vector<Real> > > & primal_,
               const std::shared_ptr<dotk::Vector<Real> > & values_);
    void value(const std::vector<std::shared_ptr<dotk::Vector<Real> > > & primal_plus_,
               const std::vector<std::shared_ptr<dotk::Vector<Real> > > & primal_minus_,
               const std::shared_ptr<dotk::Vector<Real> > & values_plus_,
               const std::shared_ptr<dotk::Vector<Real> > & values_minus_);
    void gradient(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & gradient_);
    void hessian(const dotk::Vector<Real> & primal_, const dotk::Vector<Real> & vector_, dotk::Vector<Real> & output_);

private:
    DOTk_Rosenbrock(const dotk::DOTk_Rosenbrock&);
    dotk::DOTk_Rosenbrock operator=(const dotk::DOTk_Rosenbrock&);
};

}

#endif /* DOTK_ROSENBROCK_HPP_ */
