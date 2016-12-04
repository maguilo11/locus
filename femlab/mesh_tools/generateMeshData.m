function [struc] = generateMeshData(mesh_file)

struc = [];
cubDegree = 3;

%%%%%%%%%% Read mesh from exodus file
mesh = exoread(mesh_file);

%%%%%%%%%%  Generate mesh in Intrepid format
[cellNodes, iIdxVertices, jIdxVertices, iIdxMix, jIdxMix, iIdx, jIdx, iVecIdx, dof] = getDOFmap(mesh);

%%%%%%%%%%  Mesh Data
mesh.dof = dof;
struc.mesh = mesh;
struc.spaceDim = size(mesh.p, 2);
struc.sideDim = struc.spaceDim-1;
struc.nVert = size(mesh.t, 2);
struc.numFields = struc.nVert; 
struc.numDof = struc.spaceDim*struc.nVert;
struc.nVertGrid = max( max(mesh.t) );
struc.numCells = size(mesh.t, 1);
struc.cubDegreeCell = cubDegree;
struc.cubDegreeSide = cubDegree;
struc.cellNodes = cellNodes;

%%%%%%%%%%  Global DOF Maps
struc.iIdxVertices = iIdxVertices;
struc.jIdxVertices = jIdxVertices;
struc.iIdxMix = iIdxMix;
struc.jIdxMix = jIdxMix;
struc.iIdxDof = iIdx;
struc.jIdxDof = jIdx;
struc.iVecIdx = iVecIdx;

%%%%%%%%%% Set number of stress components
switch struc.spaceDim
    case 2
        struc.numStress = 3;
    case 3
        struc.numStress = 6;  
    otherwise
        error('Only 2D and 3D problems can be solved');
end

end