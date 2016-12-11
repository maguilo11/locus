/*
 * DOTk_ObjectiveFunction.hpp
 *
 *  Created on: Mar 22, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_OBJECTIVEFUNCTION_HPP_
#define DOTK_OBJECTIVEFUNCTION_HPP_

#include <vector>
#include <tr1/memory>

#include "vector.hpp"
#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class DOTk_ObjectiveFunction
{
public:
    DOTk_ObjectiveFunction()
    {
    }
    virtual ~DOTk_ObjectiveFunction()
    {
    }

    virtual void value(const std::vector<std::tr1::shared_ptr<dotk::Vector<ScalarType> > > & primal_,
                       const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & fval_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::value(in, out). ABORT. ****\n");
        std::abort();
    }
    virtual void value(const std::vector<std::tr1::shared_ptr<dotk::Vector<ScalarType> > > & primal_plus_epsilon_,
                       const std::vector<std::tr1::shared_ptr<dotk::Vector<ScalarType> > > & primal_minus_epsilon_,
                       const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & value_plus_,
                       const std::tr1::shared_ptr<dotk::Vector<ScalarType> > & value_minus_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::value(in, in, out, out). ABORT. ****\n");
        std::abort();
    }

    virtual ScalarType value(const dotk::Vector<ScalarType> & primal_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::value(in). ABORT. ****\n");
        std::abort();
        return (0.);
    }
    virtual void gradient(const dotk::Vector<ScalarType> & primal_, dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::gradient. ABORT. ****\n");
        std::abort();
    }
    virtual void hessian(const dotk::Vector<ScalarType> & primal_,
                         const dotk::Vector<ScalarType> & vector_,
                         dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::hessian. ABORT. ****\n");
        std::abort();
    }

    virtual ScalarType value(const dotk::Vector<ScalarType> & state_, const dotk::Vector<ScalarType> & control_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::value. ABORT. ****\n");
        std::abort();
        return (0.);
    }
    virtual void partialDerivativeState(const dotk::Vector<ScalarType> & state_,
                                        const dotk::Vector<ScalarType> & control_,
                                        dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeState. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControl(const dotk::Vector<ScalarType> & state_,
                                          const dotk::Vector<ScalarType> & control_,
                                          dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeStateState(const dotk::Vector<ScalarType> & state_,
                                             const dotk::Vector<ScalarType> & control_,
                                             const dotk::Vector<ScalarType> & vector_,
                                             dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeStateState. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeStateControl(const dotk::Vector<ScalarType> & state_,
                                               const dotk::Vector<ScalarType> & control_,
                                               const dotk::Vector<ScalarType> & vector_,
                                               dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeStateControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControlControl(const dotk::Vector<ScalarType> & state_,
                                                 const dotk::Vector<ScalarType> & control_,
                                                 const dotk::Vector<ScalarType> & vector_,
                                                 dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeControlControl. ABORT. ****\n");
        std::abort();
    }
    virtual void partialDerivativeControlState(const dotk::Vector<ScalarType> & state_,
                                               const dotk::Vector<ScalarType> & control_,
                                               const dotk::Vector<ScalarType> & vector_,
                                               dotk::Vector<ScalarType> & output_)
    {
        std::perror("\n**** Calling Unimplemented Function DOTk_ObjectiveFunction::partialDerivativeControlState. ABORT. ****\n");
        std::abort();
    }

private:
    DOTk_ObjectiveFunction(const dotk::DOTk_ObjectiveFunction<ScalarType> &);
    dotk::DOTk_ObjectiveFunction<ScalarType> & operator=(const dotk::DOTk_ObjectiveFunction<ScalarType> &);
};

}

#endif /* DOTK_OBJECTIVEFUNCTION_HPP_ */
