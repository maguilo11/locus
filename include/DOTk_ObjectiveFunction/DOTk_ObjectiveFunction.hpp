/*
 * DOTk_ObjectiveFunction.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OBJECTIVEFUNCTION_HPP_
#define DOTK_OBJECTIVEFUNCTION_HPP_

#include <vector>

#include "vector.hpp"
#include "DOTk_Types.hpp"

namespace dotk
{

template<typename Type>
class DOTk_ObjectiveFunction
{
public:
    DOTk_ObjectiveFunction()
    {
    }
    virtual ~DOTk_ObjectiveFunction()
    {
    }

    virtual void value(const std::vector<std::tr1::shared_ptr<dotk::vector<Type> > > & primal_,
                       const std::tr1::shared_ptr<dotk::vector<Type> > & fval_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::value(in, out). ABORT. ****\n");
        std::abort();
    }
    virtual void value(const std::vector<std::tr1::shared_ptr<dotk::vector<Type> > > & primal_plus_epsilon_,
                       const std::vector<std::tr1::shared_ptr<dotk::vector<Type> > > & primal_minus_epsilon_,
                       const std::tr1::shared_ptr<dotk::vector<Type> > & value_plus_,
                       const std::tr1::shared_ptr<dotk::vector<Type> > & value_minus_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::value(in, in, out, out). ABORT. ****\n");
        std::abort();
    }

    virtual Type value(const dotk::vector<Type> & primal_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::value(in). ABORT. ****\n");
        std::abort();
        return (0.);
    }
    virtual void gradient(const dotk::vector<Type> & primal_, dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::gradient. ABORT. ****\n");
        std::abort();
    }
    virtual void hessian(const dotk::vector<Type> & primal_,
                         const dotk::vector<Type> & vector_,
                         dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::hessian. ABORT. ****\n");
        std::abort();
    }

    virtual Type value(const dotk::vector<Type> & state_, const dotk::vector<Type> & control_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::value. ABORT. ****\n");
        std::abort();
        return (0.);
    }
    virtual void partialDerivativeState(const dotk::vector<Type> & state_,
                                        const dotk::vector<Type> & control_,
                                        dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeState. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControl(const dotk::vector<Type> & state_,
                                          const dotk::vector<Type> & control_,
                                          dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeStateState(const dotk::vector<Type> & state_,
                                             const dotk::vector<Type> & control_,
                                             const dotk::vector<Type> & vector_,
                                             dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeStateState. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeStateControl(const dotk::vector<Type> & state_,
                                               const dotk::vector<Type> & control_,
                                               const dotk::vector<Type> & vector_,
                                               dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeStateControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControlControl(const dotk::vector<Type> & state_,
                                                 const dotk::vector<Type> & control_,
                                                 const dotk::vector<Type> & vector_,
                                                 dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeControlControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControlState(const dotk::vector<Type> & state_,
                                               const dotk::vector<Type> & control_,
                                               const dotk::vector<Type> & vector_,
                                               dotk::vector<Type> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeControlState. ABORT. ****\n");
        std::abort();
    }

private:
    DOTk_ObjectiveFunction(const dotk::DOTk_ObjectiveFunction<Type> &);
    dotk::DOTk_ObjectiveFunction<Type> & operator=(const dotk::DOTk_ObjectiveFunction<Type> &);
};

}

#endif /* DOTK_OBJECTIVEFUNCTION_HPP_ */
