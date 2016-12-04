function [usr_par] = getCubature(usr_par)
%
%  generateProbSpecificData(usr_par)
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
%           Denis Ridzal
%           Sandia National Laboratories
%           August 16, 2011

spaceDim      = usr_par.spaceDim;
sideDim       = usr_par.sideDim;
cellType      = usr_par.cellType;
sideType      = usr_par.sideType;
cubDegree     = usr_par.cubDegree;
cubDegreeSide = usr_par.cubDegreeSide;

%%%%%%%%%%% evaluate cell cubature points and weights
numCubPoints = intrepid_getNumCubaturePoints(cellType, cubDegree);
cubPoints    = zeros(spaceDim, numCubPoints);
cubWeights   = zeros(1, numCubPoints);
intrepid_getCubature(cubPoints, cubWeights, cellType, cubDegree);
usr_par.numCubPoints = numCubPoints;
usr_par.cubPoints    = cubPoints;
usr_par.cubWeights   = cubWeights;

%%%%%%%%%%% evaluate side (parent subcell) cubature points and weights
numCubPointsSide = intrepid_getNumCubaturePoints(sideType, cubDegreeSide);
cubPointsSide    = zeros(sideDim, numCubPointsSide);
cubWeightsSide   = zeros(1, numCubPointsSide);
intrepid_getCubature(cubPointsSide, cubWeightsSide, ...
    sideType, cubDegreeSide);
usr_par.numCubPointsSide = numCubPointsSide;
usr_par.cubPointsSide    = cubPointsSide;
usr_par.cubWeightsSide   = cubWeightsSide;

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
usr_par.cubPointsSide0Ref = cubPointsSide0Ref;
usr_par.cubPointsSide1Ref = cubPointsSide1Ref;
usr_par.cubPointsSide2Ref = cubPointsSide2Ref;

end
