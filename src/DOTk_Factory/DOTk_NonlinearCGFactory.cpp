/*
 * DOTk_NonlinearCGFactory.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Daniels.hpp"
#include "DOTk_DaiLiao.hpp"
#include "DOTk_DaiYuan.hpp"
#include "DOTk_LiuStorey.hpp"
#include "DOTk_HagerZhang.hpp"
#include "DOTk_PerryShanno.hpp"
#include "DOTk_PolakRibiere.hpp"
#include "DOTk_DaiYuanHybrid.hpp"
#include "DOTk_FletcherReeves.hpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_HestenesStiefel.hpp"
#include "DOTk_ConjugateDescent.hpp"
#include "DOTk_NonlinearCGFactory.hpp"

namespace dotk
{

DOTk_NonlinearCGFactory::DOTk_NonlinearCGFactory()
{
}

DOTk_NonlinearCGFactory::~DOTk_NonlinearCGFactory()
{
}

void DOTk_NonlinearCGFactory::buildDanielsNlcg(const std::shared_ptr<dotk::DOTk_LinearOperator> & hessian_,
                                               std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_Daniels(hessian_));
}

void DOTk_NonlinearCGFactory::buildFletcherReevesNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_FletcherReeves);
}

void DOTk_NonlinearCGFactory::buildPolakRibiereNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_PolakRibiere);
}

void DOTk_NonlinearCGFactory::buildHestenesStiefelNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_HestenesStiefel);
}

void DOTk_NonlinearCGFactory::buildConjugateDescentNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_ConjugateDescent);
}

void DOTk_NonlinearCGFactory::buildHagerZhangNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_HagerZhang);
}

void DOTk_NonlinearCGFactory::buildDaiLiaoNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_DaiLiao);
}

void DOTk_NonlinearCGFactory::buildDaiYuanNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_DaiYuan);
}

void DOTk_NonlinearCGFactory::buildDaiYuanHybridNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_DaiYuanHybrid);
}

void DOTk_NonlinearCGFactory::buildPerryShannoNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_PerryShanno);
}

void DOTk_NonlinearCGFactory::buildLiuStoreyNlcg(std::shared_ptr<dotk::DOTk_DescentDirection> & output_)
{
    output_.reset(new dotk::DOTk_LiuStorey);
}

}
