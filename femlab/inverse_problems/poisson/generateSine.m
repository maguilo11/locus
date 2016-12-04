function force = generateSine(struc)

%%%%%%%%%%% get computational mesh cubature points physical frame
cubPointsPhysCoord=struc.cubPointsPhysCoord;

%%%% Find the frequencies
%freqs=10*pi; % CME
freqs=8*pi; % L2
amplitude = 1e2;

clear force;
%%%% build right hand side
nVert = size(cubPointsPhysCoord,2);
numCells = size(cubPointsPhysCoord,3);
cell_force = zeros(nVert,numCells);
cell_force(:,:) = amplitude * ...
    sin(freqs*cubPointsPhysCoord(1,:,:)) .* cos(freqs*cubPointsPhysCoord(2,:,:));

%%%% integrate right hand side
integrated_force = zeros(struc.numFields, struc.numCells);
intrepid_integrate(integrated_force, ...
                   cell_force, ...
                   struc.weighted_transformed_val_at_cub_points, ...
                   'COMP_BLAS');

%%%% build global rhs vector
force_vector = reshape(integrated_force, 1, numel(integrated_force));
sparse_force = sparse(struc.iVecIdx, struc.iVecIdx, force_vector);
force = spdiags(sparse_force,0);
