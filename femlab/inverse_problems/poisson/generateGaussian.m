function force = generateGaussian(struc)

%%%%%%%%%%% get computational mesh cubature points physical frame
cubPointsPhysCoord=struc.cubPointsPhysCoord;

% Gaussian parameters
center = [0.25 0.25 0];
widths = [0.5 0.5 0.2];
ampl   = 1e1;

clear force;

%%%%%%%%%%% build right hand side
nVert = size(cubPointsPhysCoord,2);
numCells = size(cubPointsPhysCoord,3);
exponent = zeros(nVert,numCells);
for j=1:size(struc.mesh.p,2)
    exponent = exponent - (0.5/widths(j)^2) * (squeeze(cubPointsPhysCoord(j,:,:))-center(j)).^2;
end
force_at_nodes = ampl*exp(exponent);

%%%%%%%%%%% integrate right hand side
integrated_force = zeros(struc.numFields, struc.numCells);
intrepid_integrate(integrated_force, force_at_nodes, ...
    struc.weighted_transformed_val_at_cub_points, 'COMP_BLAS');

%%%% build global rhs vector
force_vector = reshape(integrated_force, 1, numel(integrated_force));
sparse_force = sparse(struc.iVecIdx, struc.iVecIdx, force_vector);
force = spdiags(sparse_force,0);

end