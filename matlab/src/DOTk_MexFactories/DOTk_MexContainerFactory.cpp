/*
 * DOTk_MexContainerFactory.cpp
 *
 *  Created on: Apr 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mpi.h>
#include "DOTk_Dual.hpp"
#include "DOTk_State.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Control.hpp"
#include "DOTk_VectorTypes.hpp"
#include "DOTk_MexAlgorithmParser.hpp"

namespace dotk
{

namespace mex
{

template<typename Type>
void buildContainer(const mxArray* options_, const size_t & dimension_, Type & variable_);
void allocatePrimalVariable(const mxArray* options_, dotk::DOTk_Primal & primal_);

void buildDualContainer(const mxArray* options_, dotk::DOTk_Dual & dual_)
{
    size_t dimension = 0;
    dotk::mex::parseNumberDuals(options_, dimension);
    dotk::mex::buildContainer(options_, dimension, dual_);
    dotk::mex::parseDualData(options_, *dual_.data());
}

void buildStateContainer(const mxArray* options_, dotk::DOTk_State & state_)
{
    size_t dimension = 0;
    dotk::mex::parseNumberStates(options_, dimension);
    dotk::mex::buildContainer(options_, dimension, state_);
    dotk::mex::parseStateData(options_, *state_.data());
}

void buildPrimalContainer(const mxArray* options_, dotk::DOTk_Primal & primal_)
{
    dotk::mex::allocatePrimalVariable(options_, primal_);
    dotk::mex::parseDualData(options_, *primal_.dual());
    dotk::mex::parseStateData(options_, *primal_.state());
    dotk::mex::parseControlData(options_, *primal_.control());
}

void buildControlContainer(const mxArray* options_, dotk::DOTk_Control & control_)
{
    size_t dimension = 0;
    dotk::mex::parseNumberControls(options_, dimension);
    dotk::mex::buildContainer(options_, dimension, control_);
    dotk::mex::parseControlData(options_, *control_.data());
}

void buildDualContainer(const mxArray* options_, dotk::DOTk_Primal & primal_)
{
    size_t number_duals = 0;
    dotk::mex::parseNumberDuals(options_, number_duals);
    dotk::types::container_t type = dotk::types::UNDEFINED_DOTK_CONTAINER;
    dotk::mex::parseContainerType(options_, type);
    switch(type)
    {
        case dotk::types::SERIAL_VECTOR:
        {
            primal_.allocateSerialDualVector(number_duals);
            break;
        }
        case dotk::types::SERIAL_ARRAY:
        {
            primal_.allocateSerialDualArray(number_duals);
            break;
        }
        case dotk::types::OMP_VECTOR:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateOmpDualVector(number_duals, thread_count);
            break;
        }
        case dotk::types::OMP_ARRAY:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateOmpDualArray(number_duals, thread_count);
            break;
        }
        case dotk::types::MPI_VECTOR:
        {
            primal_.allocateMpiDualVector(MPI_COMM_WORLD, number_duals);
            break;
        }
        case dotk::types::MPI_ARRAY:
        {
            primal_.allocateMpiDualArray(MPI_COMM_WORLD, number_duals);
            break;
        }
        case dotk::types::MPIx_VECTOR:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateMpixDualVector(MPI_COMM_WORLD, thread_count, number_duals);
            break;
        }
        case dotk::types::MPIx_ARRAY:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateMpixDualArray(MPI_COMM_WORLD, thread_count, number_duals);
            break;
        }
        case dotk::types::USER_DEFINED_CONTAINER:
        default:
        {
            primal_.allocateSerialDualArray(number_duals);
            std::string msg(" DOTk/MEX WARNING: Invalid Container Type. Default = SERIAL C ARRAY. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
    dotk::mex::parseDualData(options_, *primal_.dual());
}

void buildStateContainer(const mxArray* options_, dotk::DOTk_Primal & primal_)
{
     size_t number_states = 0;
     dotk::mex::parseNumberStates(options_, number_states);
     dotk::types::container_t type = dotk::types::UNDEFINED_DOTK_CONTAINER;
     dotk::mex::parseContainerType(options_, type);

     switch(type)
     {
         case dotk::types::SERIAL_VECTOR:
         {
             primal_.allocateSerialStateVector(number_states);
             break;
         }
         case dotk::types::SERIAL_ARRAY:
         {
             primal_.allocateSerialStateArray(number_states);
             break;
         }
         case dotk::types::OMP_VECTOR:
         {
             size_t thread_count = 0;
             dotk::mex::parseThreadCount(options_, thread_count);
             primal_.allocateOmpStateVector(number_states, thread_count);
             break;
         }
         case dotk::types::OMP_ARRAY:
         {
             size_t thread_count = 0;
             dotk::mex::parseThreadCount(options_, thread_count);
             primal_.allocateOmpStateArray(number_states, thread_count);
             break;
         }
         case dotk::types::MPI_VECTOR:
         {
             primal_.allocateMpiStateVector(MPI_COMM_WORLD, number_states);
             break;
         }
         case dotk::types::MPI_ARRAY:
         {
             primal_.allocateMpiStateArray(MPI_COMM_WORLD, number_states);
             break;
         }
         case dotk::types::MPIx_VECTOR:
         {
             size_t thread_count = 0;
             dotk::mex::parseThreadCount(options_, thread_count);
             primal_.allocateMpixStateVector(MPI_COMM_WORLD, thread_count, number_states);
             break;
         }
         case dotk::types::MPIx_ARRAY:
         {
             size_t thread_count = 0;
             dotk::mex::parseThreadCount(options_, thread_count);
             primal_.allocateMpixStateArray(MPI_COMM_WORLD, thread_count, number_states);
             break;
         }
         case dotk::types::USER_DEFINED_CONTAINER:
         default:
         {
             primal_.allocateSerialStateArray(number_states);
             std::string msg(" DOTk/MEX WARNING: Invalid Container Type. Default = SERIAL C ARRAY. \n");
             mexWarnMsgTxt(msg.c_str());
             break;
         }
     }
    dotk::mex::parseStateData(options_, *primal_.state());
}

void buildControlContainer(const mxArray* options_, dotk::DOTk_Primal & primal_)
{
    size_t number_controls = 0;
    dotk::mex::parseNumberControls(options_, number_controls);
    dotk::types::container_t type = dotk::types::UNDEFINED_DOTK_CONTAINER;
    dotk::mex::parseContainerType(options_, type);
    switch(type)
    {
        case dotk::types::SERIAL_VECTOR:
        {
            primal_.allocateSerialControlVector(number_controls);
            break;
        }
        case dotk::types::SERIAL_ARRAY:
        {
            primal_.allocateSerialControlArray(number_controls);
            break;
        }
        case dotk::types::OMP_VECTOR:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateOmpControlVector(number_controls, thread_count);
            break;
        }
        case dotk::types::OMP_ARRAY:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateOmpControlArray(number_controls, thread_count);
            break;
        }
        case dotk::types::MPI_VECTOR:
        {
            primal_.allocateMpiControlVector(MPI_COMM_WORLD, number_controls);
            break;
        }
        case dotk::types::MPI_ARRAY:
        {
            primal_.allocateMpiControlArray(MPI_COMM_WORLD, number_controls);
            break;
        }
        case dotk::types::MPIx_VECTOR:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateMpixControlVector(MPI_COMM_WORLD, thread_count, number_controls);
            break;
        }
        case dotk::types::MPIx_ARRAY:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateMpixControlArray(MPI_COMM_WORLD, thread_count, number_controls);
            break;
        }
        case dotk::types::USER_DEFINED_CONTAINER:
        default:
        {
            primal_.allocateSerialControlArray(number_controls);
            std::string msg(" DOTk/MEX WARNING: Invalid Container Type. Default = SERIAL C ARRAY. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
    dotk::mex::parseControlData(options_, *primal_.control());
}

template<typename Type>
void buildContainer(const mxArray* options_, const size_t & dimension_, Type & variable_)
{
    dotk::types::container_t type = dotk::types::UNDEFINED_DOTK_CONTAINER;
    dotk::mex::parseContainerType(options_, type);

    switch(type)
    {
        case dotk::types::SERIAL_VECTOR:
        {
            variable_.allocateSerialVector(dimension_, 0.);
            break;
        }
        case dotk::types::SERIAL_ARRAY:
        {
            variable_.allocateSerialArray(dimension_, 0.);
            break;
        }
        case dotk::types::OMP_VECTOR:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            variable_.allocateOmpVector(dimension_, thread_count, 0.);
            break;
        }
        case dotk::types::OMP_ARRAY:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            variable_.allocateOmpArray(dimension_, thread_count, 0.);
            break;
        }
        case dotk::types::MPI_VECTOR:
        {
            variable_.allocateMpiVector(MPI_COMM_WORLD, dimension_, 0.);
            break;
        }
        case dotk::types::MPI_ARRAY:
        {
            variable_.allocateMpiArray(MPI_COMM_WORLD, dimension_, 0.);
            break;
        }
        case dotk::types::MPIx_VECTOR:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            variable_.allocateMpixVector(MPI_COMM_WORLD, thread_count, dimension_, 0.);
            break;
        }
        case dotk::types::MPIx_ARRAY:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            variable_.allocateMpixArray(MPI_COMM_WORLD, thread_count, dimension_, 0.);
            break;
        }
        case dotk::types::USER_DEFINED_CONTAINER:
        default:
        {
            variable_.allocateSerialArray(dimension_, 0.);
            std::string msg(" DOTk/MEX WARNING: Invalid Container Type. Default = SERIAL C ARRAY. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
}

void allocatePrimalVariable(const mxArray* options_, dotk::DOTk_Primal & primal_)
{
    size_t number_duals = 0;
    dotk::mex::parseNumberDuals(options_, number_duals);
    size_t number_states = 0;
    dotk::mex::parseNumberStates(options_, number_states);
    size_t number_controls = 0;
    dotk::mex::parseNumberControls(options_, number_controls);

    dotk::types::container_t type = dotk::types::UNDEFINED_DOTK_CONTAINER;
    dotk::mex::parseContainerType(options_, type);

    switch(type)
    {
        case dotk::types::SERIAL_VECTOR:
        {
            primal_.allocateSerialDualVector(number_duals);
            primal_.allocateSerialStateVector(number_states);
            primal_.allocateSerialControlVector(number_controls);
            break;
        }
        case dotk::types::SERIAL_ARRAY:
        {
            primal_.allocateSerialDualArray(number_duals);
            primal_.allocateSerialStateArray(number_states);
            primal_.allocateSerialControlArray(number_controls);
            break;
        }
        case dotk::types::OMP_VECTOR:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateOmpDualVector(number_duals, thread_count);
            primal_.allocateOmpStateVector(number_states, thread_count);
            primal_.allocateOmpControlVector(number_controls, thread_count);
            break;
        }
        case dotk::types::OMP_ARRAY:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateOmpDualArray(number_duals, thread_count);
            primal_.allocateOmpStateArray(number_states, thread_count);
            primal_.allocateOmpControlArray(number_controls, thread_count);
            break;
        }
        case dotk::types::MPI_VECTOR:
        {
            primal_.allocateMpiDualVector(MPI_COMM_WORLD, number_duals);
            primal_.allocateMpiStateVector(MPI_COMM_WORLD, number_states);
            primal_.allocateMpiControlVector(MPI_COMM_WORLD, number_controls);
            break;
        }
        case dotk::types::MPI_ARRAY:
        {
            primal_.allocateMpiDualArray(MPI_COMM_WORLD, number_duals);
            primal_.allocateMpiStateArray(MPI_COMM_WORLD, number_states);
            primal_.allocateMpiControlArray(MPI_COMM_WORLD, number_controls);
            break;
        }
        case dotk::types::MPIx_VECTOR:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateMpixDualVector(MPI_COMM_WORLD, thread_count, number_duals);
            primal_.allocateMpixStateVector(MPI_COMM_WORLD, thread_count, number_states);
            primal_.allocateMpixControlVector(MPI_COMM_WORLD, thread_count, number_controls);
            break;
        }
        case dotk::types::MPIx_ARRAY:
        {
            size_t thread_count = 0;
            dotk::mex::parseThreadCount(options_, thread_count);
            primal_.allocateMpixDualArray(MPI_COMM_WORLD, thread_count, number_duals);
            primal_.allocateMpixStateArray(MPI_COMM_WORLD, thread_count, number_states);
            primal_.allocateMpixControlArray(MPI_COMM_WORLD, thread_count, number_controls);
            break;
        }
        case dotk::types::USER_DEFINED_CONTAINER:
        default:
        {
            primal_.allocateSerialDualArray(number_duals);
            primal_.allocateSerialStateArray(number_states);
            primal_.allocateSerialControlArray(number_controls);
            std::string msg(" DOTk/MEX WARNING: Invalid Container Type. Default = SERIAL C ARRAY. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
}

}

}
