function [struc] = getCubature(struc)
%
%  getCubature(usr_par)
%
%  PURPOSE: Generates cells and subcells cubature points and weights. This 
%           data can be computed once and reused, thus speeding up the 
%           overall computation.
%
%  Input:
%           usr_par    contains all input parameters, as well as additional
%                      computed quantities
%  
%  Output:
%   
%           usr_par    contains all previous input parameters, as well as 
%                      additional computed quantities; however, additional
%                      problem-specific constant quantities were added
%
%  AUTHOR:  Miguel Aguilo
%           Sandia National Laboratories
%           August 16, 2011

spaceDim      = struc.spaceDim;
sideDim       = struc.sideDim;
cellType      = struc.cellType;
sideType      = struc.sideType;
cubDegree     = struc.cubDegree;
cubDegreeSide = struc.cubDegreeSide;

%%%%%%%%%%% evaluate cell cubature points and weights
numCubPoints = intrepid_getNumCubaturePoints(cellType, cubDegree);
cubPoints    = zeros(spaceDim, numCubPoints);
cubWeights   = zeros(1, numCubPoints);
intrepid_getCubature(cubPoints, cubWeights, cellType, cubDegree);
struc.numCubPoints = numCubPoints;
struc.cubPoints    = cubPoints;
struc.cubWeights   = cubWeights;

%%%%%%%%%%% evaluate side (parent subcell) cubature points and weights
numCubPointsSide = intrepid_getNumCubaturePoints(sideType, cubDegreeSide);
cubPointsSide    = zeros(sideDim, numCubPointsSide);
cubWeightsSide   = zeros(1, numCubPointsSide);
intrepid_getCubature(cubPointsSide,cubWeightsSide,sideType,cubDegreeSide);
struc.numCubPointsSide = numCubPointsSide;
struc.cubPointsSide    = cubPointsSide;
struc.cubWeightsSide   = cubWeightsSide;

%%%%%%%%%%% compute side cubature points and weights at reference subcell
cubPointsSide0Ref = zeros(numCubPointsSide, spaceDim);
cubPointsSide1Ref = zeros(numCubPointsSide, spaceDim);
cubPointsSide2Ref = zeros(numCubPointsSide, spaceDim);
intrepid_mapToReferenceSubcell(cubPointsSide0Ref, cubPointsSide, ...
    sideDim, 0, cellType);
intrepid_mapToReferenceSubcell(cubPointsSide1Ref, cubPointsSide, ...
    sideDim, 1, cellType);
intrepid_mapToReferenceSubcell(cubPointsSide2Ref, cubPointsSide, ...
    sideDim, 2, cellType);
struc.cubPointsSide0Ref = cubPointsSide0Ref;
struc.cubPointsSide1Ref = cubPointsSide1Ref;
struc.cubPointsSide2Ref = cubPointsSide2Ref;

end