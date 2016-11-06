/*
 * DOTk_Rosenbrock.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ROSENBROCK_HPP_
#define DOTK_ROSENBROCK_HPP_

#include <tr1/memory>

#include "DOTk_ObjectiveFunction.hpp"

namespace dotk
{

template<typename Type>
class vector;

class DOTk_Rosenbrock : public dotk::DOTk_ObjectiveFunction<Real>
{
public:
    DOTk_Rosenbrock();
    virtual ~DOTk_Rosenbrock();

    Real value(const dotk::vector<Real> & primal_);
    void value(const std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > & primal_,
               const std::tr1::shared_ptr<dotk::vector<Real> > & values_);
    void value(const std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > & primal_plus_,
               const std::vector<std::tr1::shared_ptr<dotk::vector<Real> > > & primal_minus_,
               const std::tr1::shared_ptr<dotk::vector<Real> > & values_plus_,
               const std::tr1::shared_ptr<dotk::vector<Real> > & values_minus_);
    void gradient(const dotk::vector<Real> & primal_, dotk::vector<Real> & gradient_);
    void hessian(const dotk::vector<Real> & primal_, const dotk::vector<Real> & vector_, dotk::vector<Real> & output_);

private:
    DOTk_Rosenbrock(const dotk::DOTk_Rosenbrock&);
    dotk::DOTk_Rosenbrock operator=(const dotk::DOTk_Rosenbrock&);
};

}

#endif /* DOTK_ROSENBROCK_HPP_ */
