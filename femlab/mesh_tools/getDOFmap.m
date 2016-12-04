function [cellNodes, iIdxVertices, jIdxVertices, iIdxMix, jIdxMix, iIdx, jIdx, iVecIdx, dof] = getDOFmap(mesh)
% function [cellNodes, iIdx, jIdx, iVecIdx, iIdxMix, jIdxMix] = getDOFmap(mesh)
%
%  PURPOSE: Given a mesh, generate a degree-of-freedom map for linear elements.
%
%  INPUT:             mesh       structure array with the following fields
%
%                       - mesh.p     Real (# nodes) x dimension
%                                    array containing the coordinates of the vertices
%
%                       - mesh.t     Integer (# cells) x (# vertices per cell)   
%                                    cell connectivity array
%
%  OUTPUT:            cellNodes  Real dimension x (# vertices per cell) x (# cells)
%                                contains the coordinates of the vertices of each cell
%
%                     iIdx       index array used for matrix FEM assembly (trial functions)
%                                state-state
%
%                     jIdx       index array used for matrix FEM assembly (test functions)
%                                state-state
%
%                     iVecIdx    index array used for vector FEM assembly
%
%                     iIdxMix    index array used for matrix FEM assembly (trial functions)
%                                state-control
%
%                     jIdxMix    index array used for matrix FEM assembly (test functions)
%                                state-control
%
%  AUTHORS: Miguel Aguilo
%           (Sandia National Labs)

%%% get number of cells in the mesh and number of vertices per cell
[numCells, numVert] = size(mesh.t);

%%% get spatial dimension
dim = size(mesh.p, 2);

%%% scalar field DOFs:
iIdx_tmp = zeros(numVert*numCells,numVert);
jIdx_tmp = zeros(numCells,numVert*numVert);
for i=1:numVert
  iIdx_tmp(i:numVert:numVert*numCells,:) = mesh.t;
  for j=1:numVert
    jIdx_tmp(:, (i-1)*numVert+j) = mesh.t(:,i);
  end
end
iIdxVertices = double(reshape(iIdx_tmp', 1, numel(iIdx_tmp)));
jIdxVertices = double(reshape(jIdx_tmp', 1, numel(jIdx_tmp), 1));
clear iIdx_tmp;
clear jIdx_tmp;


%%% mixed state / parameter degrees of freedom:

iIdxMix = []; jIdxMix = [];
for i=1:dim
  iIdxMix = [iIdxMix; (dim*iIdxVertices) - (dim-i)];
  jIdxMix = [jIdxMix; jIdxVertices];
end
iIdxMix = reshape(iIdxMix,1,numel(iIdxMix));
jIdxMix = reshape(jIdxMix,1,numel(jIdxMix));


%%% vector field DOFs:
 
dof = zeros(size(mesh.t,1),dim*size(mesh.t,2));
for i=1:dim
  dof(:,i:dim:end) = (dim * mesh.t) - (dim-i);
end

iIdx_tmp = zeros(dim*numVert*numCells,dim*numVert);
jIdx_tmp = zeros(numCells,(dim*numVert)^2);
for i=1:dim*numVert
  iIdx_tmp(i:dim*numVert:numel(dof),:) = dof;
  for j=1:dim*numVert
    jIdx_tmp(:, (i-1)*(dim*numVert)+j) = dof(:,i);
  end
end
iIdx = reshape(iIdx_tmp', 1, numel(iIdx_tmp));
jIdx = reshape(jIdx_tmp', 1, numel(jIdx_tmp), 1);
clear iIdx_tmp;
clear jIdx_tmp;

%%% get rhs array index
iVecIdx = reshape(dof',1,numel(dof));

cellNodesAll = mesh.p(mesh.t',:)';

%%% format to match Intrepid
cellNodes    = reshape(cellNodesAll, dim, numVert, numCells);

%%% store the transpose of the dof array
dof = dof';

clear cellNodesAll;


end
