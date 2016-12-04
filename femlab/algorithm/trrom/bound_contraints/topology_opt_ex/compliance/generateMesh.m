function [usr_par] = generateMesh(usr_par, coarse_level)


%%%%%%%%%% Generate computational mesh on [xmin,xmax]x[ymin,ymax]
xint_original = usr_par.DomainSpecs.nx;
yint_original = usr_par.DomainSpecs.ny;
usr_par.DomainSpecs.nxint = xint_original*(2^coarse_level);
usr_par.DomainSpecs.nyint = yint_original*(2^coarse_level);
[mesh] = RectGrid(usr_par.DomainSpecs);
usr_par.DomainSpecs.xint_original = xint_original*(2^coarse_level);
usr_par.DomainSpecs.yint_original = yint_original*(2^coarse_level);

% Get the number of vertices in the grid and the number of working cells
nVertGrid = max( max(mesh.t(:,1:3)) );
numCells  = size(mesh.t, 1);

% Generate mesh in Intrepid format
[~, iIdx, jIdx, ~] = generateIntrepidMesh(mesh, usr_par.nVert, numCells);
usr_par.iIdxVertices = iIdx;
usr_par.jIdxVertices = jIdx;
usr_par.iIdxMix      = [(2*iIdx)-1; 2*iIdx];
usr_par.iIdxMix      = reshape(usr_par.iIdxMix,1,numel(usr_par.iIdxMix));
usr_par.jIdxMix      = [jIdx; jIdx];
usr_par.jIdxMix      = reshape(usr_par.jIdxMix,1,numel(usr_par.jIdxMix));

% Generate mesh data structure for elasticity
[cellNodes, iIdx, jIdx, iVecIdx] = ...
    generateIntrepidMeshElasticity(mesh, usr_par.nVert, numCells);

usr_par.mesh       = mesh;
usr_par.numCells   = numCells;
usr_par.nVertGrid  = nVertGrid;
usr_par.cellNodes  = cellNodes;
usr_par.iIdxDof    = iIdx;
usr_par.jIdxDof    = jIdx;
usr_par.iVecIdxDof = iVecIdx;

end