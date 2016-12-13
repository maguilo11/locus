function [struc] = generateMesh(struc)

struc.spaceDim = 2;                                                                     % physical spatial dimensions for cell (element)
struc.sideDim = 1;                                                                        % physical spatial dimensions for subcell (side)
struc.cellType  = 'Triangle';                                                      % cell (element) topology 
struc.nVert = 3;                                                                             % number of cell vertices
struc.cubDegree = 3;                                                                     % max. degree of the polynomial that can be represented by the basis
struc.numFields = 3;                                                                     % number of fields (i.e. number of basis functions)
struc.sideType = 'Line';                                                              % subcell (side) topology
struc.nVertSide = 2;                                                                        % number of subcell vertices
struc.cubDegreeSide = 2;                                                                % subcells max. degree of the polynomial that can be represented by the basis
struc.numFieldsSide = 2;                                                                % subcells number of fields (i.e. number of basis functions)
struc.numSides = 3;                                                                       % number of sides for one cell
struc.numDof = struc.spaceDim*struc.numFields; % number of degrees of freedom per element

%%%%%%%%%% Generate computational mesh on [xmin,xmax]x[ymin,ymax]
[mesh] = RectGrid(struc.DomainSpecs);

% Get the number of vertices in the grid and the number of working cells
nVertGrid = max( max(mesh.t(:,1:3)) );
numCells  = size(mesh.t, 1);

% Generate mesh in Intrepid format
[~, iIdx, jIdx, ~] = generateIntrepidMesh(mesh, struc.nVert, numCells);
struc.iIdxVertices = iIdx;
struc.jIdxVertices = jIdx;
struc.iIdxMix      = [(2*iIdx)-1; 2*iIdx];
struc.iIdxMix      = reshape(struc.iIdxMix,1,numel(struc.iIdxMix));
struc.jIdxMix      = [jIdx; jIdx];
struc.jIdxMix      = reshape(struc.jIdxMix,1,numel(struc.jIdxMix));

% Generate mesh data structure for elasticity
[cellNodes, iIdx, jIdx, iVecIdx] = ...
    generateIntrepidMeshElasticity(mesh, struc.nVert, numCells);

struc.mesh       = mesh;
struc.numCells   = numCells;
struc.nVertGrid  = nVertGrid;
struc.cellNodes  = cellNodes;
struc.iIdxDof    = iIdx;
struc.jIdxDof    = jIdx;
struc.iVecIdxDof = iVecIdx;

end