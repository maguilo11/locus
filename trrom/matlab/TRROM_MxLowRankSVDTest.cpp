/*
 * TRROM_MxLowRankSVDTest.cpp
 *
 *  Created on: Dec 5, 2016
 *      Author: maguilo
 */

#include <string>

#include "TRROM_MxVector.hpp"
#include "TRROM_MxMatrix.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_MxBrandLowRankSVD.hpp"

namespace trrom
{

namespace mx
{

inline void setBrandAlgorithmTestData(std::tr1::shared_ptr<trrom::Matrix<double> > & current_data_set_,
                                      std::tr1::shared_ptr<trrom::Vector<double> > & current_singular_values_,
                                      std::tr1::shared_ptr<trrom::Matrix<double> > & current_left_singular_vectors_,
                                      std::tr1::shared_ptr<trrom::Matrix<double> > & current_right_singular_vectors_)
{
    // Assign contents to current data set
    int num_rows = 10;
    int num_columns = 6;
    current_data_set_.reset(new trrom::MxMatrix(num_rows, num_columns));
    (*current_data_set_)(0, 0) = 0.075854289563063;
    (*current_data_set_)(0, 1) = 0.162182308193243;
    (*current_data_set_)(0, 2) = 0.450541598502498;
    (*current_data_set_)(0, 3) = 0.106652770180584;
    (*current_data_set_)(0, 4) = 0.431413827463545;
    (*current_data_set_)(0, 5) = 0.853031117721894;
    (*current_data_set_)(1, 0) = 0.053950118666607;
    (*current_data_set_)(1, 1) = 0.794284540683907;
    (*current_data_set_)(1, 2) = 0.083821377996932;
    (*current_data_set_)(1, 3) = 0.961898080855054;
    (*current_data_set_)(1, 4) = 0.910647594429523;
    (*current_data_set_)(1, 5) = 0.622055131485066;
    (*current_data_set_)(2, 0) = 0.530797553008973;
    (*current_data_set_)(2, 1) = 0.311215042044805;
    (*current_data_set_)(2, 2) = 0.228976968716819;
    (*current_data_set_)(2, 3) = 0.004634224134067;
    (*current_data_set_)(2, 4) = 0.181847028302853;
    (*current_data_set_)(2, 5) = 0.350952380892271;
    (*current_data_set_)(3, 0) = 0.779167230102011;
    (*current_data_set_)(3, 1) = 0.528533135506213;
    (*current_data_set_)(3, 2) = 0.913337361501670;
    (*current_data_set_)(3, 3) = 0.774910464711502;
    (*current_data_set_)(3, 4) = 0.263802916521990;
    (*current_data_set_)(3, 5) = 0.513249539867053;
    (*current_data_set_)(4, 0) = 0.934010684229183;
    (*current_data_set_)(4, 1) = 0.165648729499781;
    (*current_data_set_)(4, 2) = 0.152378018969223;
    (*current_data_set_)(4, 3) = 0.817303220653433;
    (*current_data_set_)(4, 4) = 0.145538980384717;
    (*current_data_set_)(4, 5) = 0.401808033751942;
    (*current_data_set_)(5, 0) = 0.129906208473730;
    (*current_data_set_)(5, 1) = 0.601981941401637;
    (*current_data_set_)(5, 2) = 0.825816977489547;
    (*current_data_set_)(5, 3) = 0.868694705363510;
    (*current_data_set_)(5, 4) = 0.136068558708664;
    (*current_data_set_)(5, 5) = 0.075966691690841;
    (*current_data_set_)(6, 0) = 0.568823660872193;
    (*current_data_set_)(6, 1) = 0.262971284540144;
    (*current_data_set_)(6, 2) = 0.538342435260057;
    (*current_data_set_)(6, 3) = 0.084435845510910;
    (*current_data_set_)(6, 4) = 0.869292207640089;
    (*current_data_set_)(6, 5) = 0.239916153553658;
    (*current_data_set_)(7, 0) = 0.469390641058206;
    (*current_data_set_)(7, 1) = 0.654079098476782;
    (*current_data_set_)(7, 2) = 0.996134716626886;
    (*current_data_set_)(7, 3) = 0.399782649098897;
    (*current_data_set_)(7, 4) = 0.579704587365570;
    (*current_data_set_)(7, 5) = 0.123318934835166;
    (*current_data_set_)(8, 0) = 0.011902069501241;
    (*current_data_set_)(8, 1) = 0.689214503140008;
    (*current_data_set_)(8, 2) = 0.078175528753183;
    (*current_data_set_)(8, 3) = 0.259870402850654;
    (*current_data_set_)(8, 4) = 0.549860201836332;
    (*current_data_set_)(8, 5) = 0.183907788282417;
    (*current_data_set_)(9, 0) = 0.337122644398882;
    (*current_data_set_)(9, 1) = 0.748151592823710;
    (*current_data_set_)(9, 2) = 0.442678269775446;
    (*current_data_set_)(9, 3) = 0.800068480224308;
    (*current_data_set_)(9, 4) = 0.144954798223727;
    (*current_data_set_)(9, 5) = 0.239952525664903;

    // Assign contents to current singular values
    int num_singular_values = 3;
    current_singular_values_.reset(new trrom::MxVector(num_singular_values));
    (*current_singular_values_)[0] = 2.76159072774184;
    (*current_singular_values_)[1] = 0.951101592150856;
    (*current_singular_values_)[2] = 0.718247784809067;

    // Assign contents to current left singular vectors
    current_left_singular_vectors_.reset(new trrom::MxMatrix(num_rows, num_singular_values));
    (*current_left_singular_vectors_)(0, 0) = -0.305337446414953;
    (*current_left_singular_vectors_)(0, 1) = 0.149929755643348;
    (*current_left_singular_vectors_)(0, 2) = 0.454738659588238;
    (*current_left_singular_vectors_)(1, 0) = -0.161337182800233;
    (*current_left_singular_vectors_)(1, 1) = 0.297250343025266;
    (*current_left_singular_vectors_)(1, 2) = -0.035293756680850;
    (*current_left_singular_vectors_)(2, 0) = -0.406496477165708;
    (*current_left_singular_vectors_)(2, 1) = -0.306927516771071;
    (*current_left_singular_vectors_)(2, 2) = -0.626176494523147;
    (*current_left_singular_vectors_)(3, 0) = -0.300220545037820;
    (*current_left_singular_vectors_)(3, 1) = -0.567414531313143;
    (*current_left_singular_vectors_)(3, 2) = 0.387289844252935;
    (*current_left_singular_vectors_)(4, 0) = -0.138015061324149;
    (*current_left_singular_vectors_)(4, 1) = -0.340273485747276;
    (*current_left_singular_vectors_)(4, 2) = 0.063713110634888;
    (*current_left_singular_vectors_)(5, 0) = -0.168893004368106;
    (*current_left_singular_vectors_)(5, 1) = -0.272025188535749;
    (*current_left_singular_vectors_)(5, 2) = -0.040910463837152;
    (*current_left_singular_vectors_)(6, 0) = -0.400119129029725;
    (*current_left_singular_vectors_)(6, 1) = 0.459345783984302;
    (*current_left_singular_vectors_)(6, 2) = 0.146343369003984;
    (*current_left_singular_vectors_)(7, 0) = -0.543617952688073;
    (*current_left_singular_vectors_)(7, 1) = 0.032227990793972;
    (*current_left_singular_vectors_)(7, 2) = 0.098604568591887;
    (*current_left_singular_vectors_)(8, 0) = -0.327568742239719;
    (*current_left_singular_vectors_)(8, 1) = 0.241999491250776;
    (*current_left_singular_vectors_)(8, 2) = -0.173257748931569;
    (*current_left_singular_vectors_)(9, 0) = -0.121973418343343;
    (*current_left_singular_vectors_)(9, 1) = 0.112315877446613;
    (*current_left_singular_vectors_)(9, 2) = -0.427768815809286;


    // Assign contents to current right singular vectors
    current_right_singular_vectors_.reset(new trrom::MxMatrix(num_singular_values, num_singular_values));
    (*current_right_singular_vectors_)(0, 0) = -0.570895481691234;
    (*current_right_singular_vectors_)(0, 1) = 0.590426379303292;
    (*current_right_singular_vectors_)(0, 2) = -0.570504197712286;
    (*current_right_singular_vectors_)(1, 0) = -0.547145713442969;
    (*current_right_singular_vectors_)(1, 1) = 0.244479784980209;
    (*current_right_singular_vectors_)(1, 2) = 0.800538070922936;
    (*current_right_singular_vectors_)(2, 0) = -0.612135538296477;
    (*current_right_singular_vectors_)(2, 1) = -0.76917249389121;
    (*current_right_singular_vectors_)(2, 2) = -0.183476857929403;

    (*current_right_singular_vectors_)(0, 0) = -0.613893539297580;
    (*current_right_singular_vectors_)(0, 1) = -0.788524221874089;
    (*current_right_singular_vectors_)(0, 2) = 0.036936078927717;
    (*current_right_singular_vectors_)(1, 0) = -0.596991683723300;
    (*current_right_singular_vectors_)(1, 1) = 0.494374466446217;
    (*current_right_singular_vectors_)(1, 2) = 0.631818657916366;
    (*current_right_singular_vectors_)(2, 0) = -0.516464569911539;
    (*current_right_singular_vectors_)(2, 1) = 0.365818860153330;
    (*current_right_singular_vectors_)(2, 2) = -0.774235693818238;
}

inline void setBrandAlgorithmTestGold(std::tr1::shared_ptr<trrom::Vector<double> > & gold_singular_values_,
                                      std::tr1::shared_ptr<trrom::Matrix<double> > & gold_left_singular_vectors_,
                                      std::tr1::shared_ptr<trrom::Matrix<double> > & gold_right_singular_vectors_)
{
    const int num_singular_values = 9;
    gold_singular_values_.reset(new trrom::MxVector(num_singular_values));
    (*gold_singular_values_)[0] = 4.35066187820290;
    (*gold_singular_values_)[1] = 1.71801176275607;
    (*gold_singular_values_)[2] = 1.24206362721136;
    (*gold_singular_values_)[3] = 1.04867021793580;
    (*gold_singular_values_)[4] = 0.93712419112535;
    (*gold_singular_values_)[5] = 0.56356132487280;
    (*gold_singular_values_)[6] = 0.39410132854473;
    (*gold_singular_values_)[7] = 0.26039713029619;
    (*gold_singular_values_)[8] = 0.17557989974362;

    int num_rows = 10;
    int num_columns = 9;
    gold_left_singular_vectors_.reset(new trrom::MxMatrix(num_rows, num_columns));
    (*gold_left_singular_vectors_)(0, 0) = -0.279542725299900;
    (*gold_left_singular_vectors_)(0, 1) = -0.158507411951207;
    (*gold_left_singular_vectors_)(0, 2) = -0.0638679695334556;
    (*gold_left_singular_vectors_)(0, 3) = -0.705494160202435;
    (*gold_left_singular_vectors_)(0, 4) = -0.0106391376865884;
    (*gold_left_singular_vectors_)(0, 5) = -0.00280164888061625;
    (*gold_left_singular_vectors_)(0, 6) = -0.510847914350164;
    (*gold_left_singular_vectors_)(0, 7) = -0.196101391702302;
    (*gold_left_singular_vectors_)(0, 8) = -0.302418785463393;
    (*gold_left_singular_vectors_)(1, 0) = -0.324254510062688;
    (*gold_left_singular_vectors_)(1, 1) = 0.157186924503431;
    (*gold_left_singular_vectors_)(1, 2) = -0.679104427602708;
    (*gold_left_singular_vectors_)(1, 3) = -0.0365558561323417;
    (*gold_left_singular_vectors_)(1, 4) = -0.219155315721193;
    (*gold_left_singular_vectors_)(1, 5) = -0.320135607859284;
    (*gold_left_singular_vectors_)(1, 6) = 0.0233280046871628;
    (*gold_left_singular_vectors_)(1, 7) = 0.385119515173098;
    (*gold_left_singular_vectors_)(1, 8) = 0.102785103005572;
    (*gold_left_singular_vectors_)(2, 0) = -0.281022381907279;
    (*gold_left_singular_vectors_)(2, 1) = -0.303853285801730;
    (*gold_left_singular_vectors_)(2, 2) = 0.592720001802398;
    (*gold_left_singular_vectors_)(2, 3) = 0.0727501520071106;
    (*gold_left_singular_vectors_)(2, 4) = -0.314136256587135;
    (*gold_left_singular_vectors_)(2, 5) = -0.514422896491182;
    (*gold_left_singular_vectors_)(2, 6) = -0.0958546847921467;
    (*gold_left_singular_vectors_)(2, 7) = 0.296797974607672;
    (*gold_left_singular_vectors_)(2, 8) = 0.105870517734808;
    (*gold_left_singular_vectors_)(3, 0) = -0.404612128782015;
    (*gold_left_singular_vectors_)(3, 1) = 0.394254942540497;
    (*gold_left_singular_vectors_)(3, 2) = 0.225729474983771;
    (*gold_left_singular_vectors_)(3, 3) = -0.342623757325863;
    (*gold_left_singular_vectors_)(3, 4) = 0.0615043652144012;
    (*gold_left_singular_vectors_)(3, 5) = 0.233062311084353;
    (*gold_left_singular_vectors_)(3, 6) = 0.183959354903901;
    (*gold_left_singular_vectors_)(3, 7) = -0.0444596004250001;
    (*gold_left_singular_vectors_)(3, 8) = 0.644494652598095;
    (*gold_left_singular_vectors_)(4, 0) = -0.262999101837366;
    (*gold_left_singular_vectors_)(4, 1) = 0.264496359818762;
    (*gold_left_singular_vectors_)(4, 2) = 0.104748571439121;
    (*gold_left_singular_vectors_)(4, 3) = 0.0764031011404317;
    (*gold_left_singular_vectors_)(4, 4) = -0.665623585749232;
    (*gold_left_singular_vectors_)(4, 5) = 0.293495903967974;
    (*gold_left_singular_vectors_)(4, 6) = 0.286360389113589;
    (*gold_left_singular_vectors_)(4, 7) = -0.143846445746809;
    (*gold_left_singular_vectors_)(4, 8) = -0.459749477813780;
    (*gold_left_singular_vectors_)(5, 0) = -0.261166736685368;
    (*gold_left_singular_vectors_)(5, 1) = 0.453076223653115;
    (*gold_left_singular_vectors_)(5, 2) = 0.0925026050157497;
    (*gold_left_singular_vectors_)(5, 3) = 0.135970450593229;
    (*gold_left_singular_vectors_)(5, 4) = 0.434852371868496;
    (*gold_left_singular_vectors_)(5, 5) = -0.129972100908741;
    (*gold_left_singular_vectors_)(5, 6) = -0.0650040427589379;
    (*gold_left_singular_vectors_)(5, 7) = 0.377026611876893;
    (*gold_left_singular_vectors_)(5, 8) = -0.390018761885578;
    (*gold_left_singular_vectors_)(6, 0) = -0.338492725940098;
    (*gold_left_singular_vectors_)(6, 1) = -0.525164551430374;
    (*gold_left_singular_vectors_)(6, 2) = -0.193010911384813;
    (*gold_left_singular_vectors_)(6, 3) = 0.239137634720504;
    (*gold_left_singular_vectors_)(6, 4) = -0.0318810371982139;
    (*gold_left_singular_vectors_)(6, 5) = 0.528066644687395;
    (*gold_left_singular_vectors_)(6, 6) = -0.107557599347524;
    (*gold_left_singular_vectors_)(6, 7) = 0.287399780652238;
    (*gold_left_singular_vectors_)(6, 8) = 0.135253629365896;
    (*gold_left_singular_vectors_)(7, 0) = -0.442010798707232;
    (*gold_left_singular_vectors_)(7, 1) = -0.205480514931661;
    (*gold_left_singular_vectors_)(7, 2) = 0.140584990463701;
    (*gold_left_singular_vectors_)(7, 3) = 0.114286083406620;
    (*gold_left_singular_vectors_)(7, 4) = 0.454629904361166;
    (*gold_left_singular_vectors_)(7, 5) = 0.121669641873105;
    (*gold_left_singular_vectors_)(7, 6) = 0.236656756188668;
    (*gold_left_singular_vectors_)(7, 7) = -0.115580081164259;
    (*gold_left_singular_vectors_)(7, 8) = -0.249189467085886;
    (*gold_left_singular_vectors_)(8, 0) = -0.255465912771689;
    (*gold_left_singular_vectors_)(8, 1) = -0.230243434508836;
    (*gold_left_singular_vectors_)(8, 2) = -0.233916357809177;
    (*gold_left_singular_vectors_)(8, 3) = 0.0329568701257758;
    (*gold_left_singular_vectors_)(8, 4) = 0.0834729219983025;
    (*gold_left_singular_vectors_)(8, 5) = -0.423333921495965;
    (*gold_left_singular_vectors_)(8, 6) = 0.428645484784673;
    (*gold_left_singular_vectors_)(8, 7) = -0.510315614508210;
    (*gold_left_singular_vectors_)(8, 8) = 0.0286526885542996;
    (*gold_left_singular_vectors_)(9, 0) = -0.247870989329076;
    (*gold_left_singular_vectors_)(9, 1) = 0.236929137263335;
    (*gold_left_singular_vectors_)(9, 2) = -0.0346994726727075;
    (*gold_left_singular_vectors_)(9, 3) = 0.531599817357486;
    (*gold_left_singular_vectors_)(9, 4) = -0.0506908205979331;
    (*gold_left_singular_vectors_)(9, 5) = -0.0515199119473660;
    (*gold_left_singular_vectors_)(9, 6) = -0.598265192591394;
    (*gold_left_singular_vectors_)(9, 7) = -0.451598525124862;
    (*gold_left_singular_vectors_)(9, 8) = 0.163407615332385;

    num_rows = 9;
    num_columns = 9;
    gold_right_singular_vectors_.reset(new trrom::MxMatrix(num_rows, num_columns));
    (*gold_right_singular_vectors_)(0, 0) = -0.381463050250012;
    (*gold_right_singular_vectors_)(0, 1) = -0.017236141543870;
    (*gold_right_singular_vectors_)(0, 2) = 0.608460868927007;
    (*gold_right_singular_vectors_)(0, 3) = -0.186227630875125;
    (*gold_right_singular_vectors_)(0, 4) = 0.0734619105241500;
    (*gold_right_singular_vectors_)(0, 5) = -0.0975361272757088;
    (*gold_right_singular_vectors_)(0, 6) = 0.640593110944792;
    (*gold_right_singular_vectors_)(0, 7) = 0.120772023168509;
    (*gold_right_singular_vectors_)(0, 8) = 0.0970985757358187;
    (*gold_right_singular_vectors_)(1, 0) = -0.353724140683131;
    (*gold_right_singular_vectors_)(1, 1) = -0.407899108966900;
    (*gold_right_singular_vectors_)(1, 2) = -0.194674324881089;
    (*gold_right_singular_vectors_)(1, 3) = -0.262155771320927;
    (*gold_right_singular_vectors_)(1, 4) = 0.318024211998634;
    (*gold_right_singular_vectors_)(1, 5) = 0.392537496479044;
    (*gold_right_singular_vectors_)(1, 6) = 0.0416831054605909;
    (*gold_right_singular_vectors_)(1, 7) = -0.282465578162113;
    (*gold_right_singular_vectors_)(1, 8) = -0.514902358035510;
    (*gold_right_singular_vectors_)(2, 0) = -0.297195786007758;
    (*gold_right_singular_vectors_)(2, 1) = -0.390426017042307;
    (*gold_right_singular_vectors_)(2, 2) = 0.0530930938724342;
    (*gold_right_singular_vectors_)(2, 3) = 0.383433796226165;
    (*gold_right_singular_vectors_)(2, 4) = 0.0545436139400175;
    (*gold_right_singular_vectors_)(2, 5) = -0.723158029476596;
    (*gold_right_singular_vectors_)(2, 6) = -0.190340865230632;
    (*gold_right_singular_vectors_)(2, 7) = -0.160031883106617;
    (*gold_right_singular_vectors_)(2, 8) = -0.147069532143306;
    (*gold_right_singular_vectors_)(3, 0) = -0.373612028125295;
    (*gold_right_singular_vectors_)(3, 1) = 0.664679927250780;
    (*gold_right_singular_vectors_)(3, 2) = -0.202284067986852;
    (*gold_right_singular_vectors_)(3, 3) = 0.231735366703170;
    (*gold_right_singular_vectors_)(3, 4) = -0.189217819266853;
    (*gold_right_singular_vectors_)(3, 5) = -0.0604433410143568;
    (*gold_right_singular_vectors_)(3, 6) = 0.128756832200265;
    (*gold_right_singular_vectors_)(3, 7) = 0.0851019621824018;
    (*gold_right_singular_vectors_)(3, 8) = -0.510604618076469;
    (*gold_right_singular_vectors_)(4, 0) = -0.320947849032051;
    (*gold_right_singular_vectors_)(4, 1) = -0.205036953023348;
    (*gold_right_singular_vectors_)(4, 2) = 0.109020198835015;
    (*gold_right_singular_vectors_)(4, 3) = 0.487366245518645;
    (*gold_right_singular_vectors_)(4, 4) = -0.526193134366675;
    (*gold_right_singular_vectors_)(4, 5) = 0.472845727956091;
    (*gold_right_singular_vectors_)(4, 6) = -0.0230523448597342;
    (*gold_right_singular_vectors_)(4, 7) = -0.212473411887903;
    (*gold_right_singular_vectors_)(4, 8) = 0.243726089304913;
    (*gold_right_singular_vectors_)(5, 0) = -0.380017404543643;
    (*gold_right_singular_vectors_)(5, 1) = 0.080124464647595;
    (*gold_right_singular_vectors_)(5, 2) = 0.105400754246182;
    (*gold_right_singular_vectors_)(5, 3) = -0.632713678352671;
    (*gold_right_singular_vectors_)(5, 4) = -0.410557117236113;
    (*gold_right_singular_vectors_)(5, 5) = -0.131897953498338;
    (*gold_right_singular_vectors_)(5, 6) = -0.492566625051734;
    (*gold_right_singular_vectors_)(5, 7) = -0.0155590754371503;
    (*gold_right_singular_vectors_)(5, 8) = 0.0944073251661460;
    (*gold_right_singular_vectors_)(6, 0) = -0.276519866797561;
    (*gold_right_singular_vectors_)(6, 1) = 0.195274770595987;
    (*gold_right_singular_vectors_)(6, 2) = 0.328698171671379;
    (*gold_right_singular_vectors_)(6, 3) = 0.235728754877503;
    (*gold_right_singular_vectors_)(6, 4) = 0.557181114413292;
    (*gold_right_singular_vectors_)(6, 5) = 0.242942326933166;
    (*gold_right_singular_vectors_)(6, 6) = -0.503080046848584;
    (*gold_right_singular_vectors_)(6, 7) = 0.291359242469488;
    (*gold_right_singular_vectors_)(6, 8) = 0.119760106523963;
    (*gold_right_singular_vectors_)(7, 0) = -0.315910492846118;
    (*gold_right_singular_vectors_)(7, 1) = -0.258579594055188;
    (*gold_right_singular_vectors_)(7, 2) = -0.540029053898322;
    (*gold_right_singular_vectors_)(7, 3) = -0.0151353477468278;
    (*gold_right_singular_vectors_)(7, 4) = -0.00894185061743602;
    (*gold_right_singular_vectors_)(7, 5) = -0.0185233360232932;
    (*gold_right_singular_vectors_)(7, 6) = 0.145780177159543;
    (*gold_right_singular_vectors_)(7, 7) = 0.673913858457982;
    (*gold_right_singular_vectors_)(7, 8) = 0.256206707363320;
    (*gold_right_singular_vectors_)(8, 0) = -0.278873564600285;
    (*gold_right_singular_vectors_)(8, 1) = 0.292632239204508;
    (*gold_right_singular_vectors_)(8, 2) = -0.354218813128713;
    (*gold_right_singular_vectors_)(8, 3) = -0.0471305009445482;
    (*gold_right_singular_vectors_)(8, 4) = 0.314194919143982;
    (*gold_right_singular_vectors_)(8, 5) = -0.0971897313920742;
    (*gold_right_singular_vectors_)(8, 6) = 0.132666249686994;
    (*gold_right_singular_vectors_)(8, 7) = -0.536970795866515;
    (*gold_right_singular_vectors_)(8, 8) = 0.542956163357781;
}

}

}

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX INTERFACE FOR BRAND ALGORITHM\n");
    mexPrintf("%s", msg.c_str());
    if(!(nInput == 0 && nOutput == 0))
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT AGUMENTS. FUNCTION TAKES NO INPUS AND RETURNS NO OUTPUTS.\n");
        mexErrMsgTxt(error.c_str());
    }

    trrom::MxBrandLowRankSVD thin_svd;

    // **** TEST 1: solve ****
    std::tr1::shared_ptr<trrom::Matrix<double> > data;
    std::tr1::shared_ptr<trrom::Vector<double> > singular_values;
    std::tr1::shared_ptr<trrom::Matrix<double> > left_singular_vectors;
    std::tr1::shared_ptr<trrom::Matrix<double> > right_singular_vectors;
    trrom::mx::setBrandAlgorithmTestData(data, singular_values, left_singular_vectors, right_singular_vectors);
    thin_svd.solve(data, singular_values, left_singular_vectors, right_singular_vectors);

    // SET GOLD VALUES
    std::tr1::shared_ptr<trrom::Vector<double> > gold_singular_values;
    std::tr1::shared_ptr<trrom::Matrix<double> > gold_left_singular_vectors;
    std::tr1::shared_ptr<trrom::Matrix<double> > gold_right_singular_vectors;
    trrom::mx::setBrandAlgorithmTestGold(gold_singular_values, gold_left_singular_vectors, gold_right_singular_vectors);

    // ASSERT TEST 1 RESULTS
    msg.assign("singular values");
    bool did_test_pass = trrom::mx::checkResults(*gold_singular_values, *singular_values);
    trrom::mx::assert_test(msg, did_test_pass);

    msg.assign("left singular vectors");
    did_test_pass = trrom::mx::checkResults(*gold_left_singular_vectors, *left_singular_vectors);
    trrom::mx::assert_test(msg, did_test_pass);

    msg.assign("right singular vectors");
    did_test_pass = trrom::mx::checkResults(*gold_right_singular_vectors, *right_singular_vectors);
    trrom::mx::assert_test(msg, did_test_pass);
}
