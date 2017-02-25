function [Operators] = objectiveFunction()
Operators.evaluate=@(state,control)evaluate(state,control);
Operators.gradient=@(state,control)gradient(state,control);
Operators.firstDerivativeWrtState=...
    @(state,control)firstDerivativeWrtState(state,control);
Operators.firstDerivativeWrtControl=...
    @(state,control)firstDerivativeWrtControl(state,control);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = evaluate(state,control)

global GLB_INVP;

%%%%%%%%%%%%%%%%%%%%%%%%% evaluate stress measure %%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% evaluate shear modulus at the cubature points
shear_at_dof = GLB_INVP.G( GLB_INVP.mesh.t');
shear_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(shear_at_cub_points, shear_at_dof, GLB_INVP.transformed_val_at_cub_points);
    
%%%%%%%%%%% evaluate bulk modulus at the cubature points
bulk_at_dof = GLB_INVP.B( GLB_INVP.mesh.t');
bulk_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(bulk_at_cub_points, bulk_at_dof, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% evaluate strain at the cubature points
state_at_dof = state( GLB_INVP.mesh.dof );
strain_at_cub_points = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(strain_at_cub_points, state_at_dof, GLB_INVP.Bmat);

%%%%%%%%%%% compute deviatoric strain 
dev_strain = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_tensorMultiplyDataData(dev_strain, GLB_INVP.Ddev, strain_at_cub_points);

%%%%%%%%%%% compute volumetric strain 
vol_strain = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_tensorMultiplyDataData(vol_strain, GLB_INVP.Dvol, strain_at_cub_points);

%%%%%%%%%%% compute deviatoric stress
dev_stress = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_scalarMultiplyDataData(dev_stress, shear_at_cub_points, dev_strain);

%%%%%%%%%%% compute volumetric stress
vol_stress = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_scalarMultiplyDataData(vol_stress, bulk_at_cub_points, vol_strain);

%%%%%%%%%%% compute average stress per cell
stress = dev_stress + vol_stress;
elem_avg_stress = zeros(GLB_INVP.numStress, GLB_INVP.numCells);
elem_avg_stress(:,:) = sum(stress,2) ./ size(stress,2);

%%%%%%%%%%% compute cell objective measure
filtered_control = GLB_INVP.Filter*control;
control_at_dof = filtered_control( GLB_INVP.mesh.t');
elem_stress_measure = zeros(GLB_INVP.numCells,1);
for cell=1:GLB_INVP.numCells
    elem_mass_volume = sum( GLB_INVP.CellMassMatrices(:,:,cell)*control_at_dof(:,cell) );
    elem_penalty = elem_mass_volume / GLB_INVP.ElemVolume(cell);
    elem_penalty = GLB_INVP.MinStressValue + ...
        ((1-GLB_INVP.MinStressValue) * elem_penalty^GLB_INVP.StressPower);
    elem_von_mises = (elem_penalty * elem_avg_stress(1,cell))^2 + ...
        (elem_penalty * elem_avg_stress(2,cell))^2 - ...
        ((elem_penalty * elem_avg_stress(1,cell)) * ...
        (elem_penalty * elem_avg_stress(2,cell))) + ...
        (3 * (elem_penalty * elem_avg_stress(3,cell))^2);
    cell_stress_ratio = sqrt(elem_von_mises)/GLB_INVP.StressNormFactor(cell);
    elem_stress_measure(cell) = GLB_INVP.ElemVolume(cell) * (cell_stress_ratio^GLB_INVP.PowerKS);
end
stress_measure = ((GLB_INVP.constant/GLB_INVP.OriginalVolume) * ...
    sum(elem_stress_measure))^(1/GLB_INVP.PowerKS);

%%%%%%%%%%%%%%%%%%%%%%%%% evaluate stress measure %%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% evaluate perimeter control term %%%%%%%%%%%%%%%%%%%%%
switch GLB_INVP.reg
    case{'Tikhonov'}
        perimeter_control = 0.5 * GLB_INVP.beta * (control'*(GLB_INVP.Ms*control));
    case{'TV'}
        perimeter_control = 0.5*GLB_INVP.beta * ...
            sqrt( (control' * (GLB_INVP.Ss * control)) + GLB_INVP.gamma );
end
%%%%%%%%%%%%%%%%%%%%% evaluate perimeter control term %%%%%%%%%%%%%%%%%%%%%

output = stress_measure + perimeter_control;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = gradient(state,control)

global GLB_INVP;

%%%%%%%%%%%%%%% compute first derivative Fval wrt Control %%%%%%%%%%%%%%%%%
[f_z] = firstDerivativeWrtControl(state,control);
%%%%%%%%%%%%%%% compute first derivative Fval wrt Control %%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% solve adjoint problem %%%%%%%%%%%%%%%%%%%%%%%%%%%
equality = equalityConstraint;
[f_u] = firstDerivativeWrtState(state,control);
[dual] = equality.applyInverseAdjointJacobianWrtState(state,control,-f_u);
[g_z_times_dual] = equality.adjointFirstDerivativeWrtControl(state,control,dual);
%%%%%%%%%%%%%%%%%%%%%%%%% solve adjoint problem %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% objective gradient %%%%%%%%%%%%%%%%%%%%%%%%%%%%
f_grad = f_z + g_z_times_dual;
%%%%%%%%%%%%%%%%%%%%%%%%%%% objective gradient %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%% perimeter control gradient %%%%%%%%%%%%%%%%%%%%%%%%
switch GLB_INVP.reg
    case{'Tikhonov'}
        perimeter_control_grad = GLB_INVP.beta * (GLB_INVP.Ms*control);
    case{'TV'}
        perimeter_control_grad = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( control' * (GLB_INVP.Ss * control) + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.Ss * control);
end
%%%%%%%%%%%%%%%%%%%%%%% perimeter control gradient %%%%%%%%%%%%%%%%%%%%%%%%

output = f_grad + perimeter_control_grad;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtControl(state,control)

global GLB_INVP;

%%%%%%%%%%% evaluate shear modulus at the cubature points
shear_at_dof = GLB_INVP.G( GLB_INVP.mesh.t');
shear_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(shear_at_cub_points, shear_at_dof, GLB_INVP.transformed_val_at_cub_points);
    
%%%%%%%%%%% evaluate bulk modulus at the cubature points
bulk_at_dof = GLB_INVP.B( GLB_INVP.mesh.t');
bulk_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(bulk_at_cub_points, bulk_at_dof, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% evaluate strain at the cubature points
state_at_dof = state( GLB_INVP.mesh.dof );
strain_at_cub_points = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(strain_at_cub_points, state_at_dof, GLB_INVP.Bmat);

%%%%%%%%%%% compute deviatoric strain 
dev_strain = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_tensorMultiplyDataData(dev_strain, GLB_INVP.Ddev, strain_at_cub_points);

%%%%%%%%%%% compute volumetric strain 
vol_strain = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_tensorMultiplyDataData(vol_strain, GLB_INVP.Dvol, strain_at_cub_points);

%%%%%%%%%%% compute deviatoric stress
dev_stress = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_scalarMultiplyDataData(dev_stress, shear_at_cub_points, dev_strain);

%%%%%%%%%%% compute volumetric stress
vol_stress = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_scalarMultiplyDataData(vol_stress, bulk_at_cub_points, vol_strain);

%%%%%%%%%%% compute average stress per cell
stress = dev_stress + vol_stress;
elem_avg_stress = zeros(GLB_INVP.numStress, GLB_INVP.numCells);
elem_avg_stress(:,:) = sum(stress,2) ./ size(stress,2);

%%%%%%%%%%% compute cell objective measure
filtered_control = GLB_INVP.Filter*control;
filtered_control_at_dof = filtered_control( GLB_INVP.mesh.t');
elem_stress_measure = zeros(GLB_INVP.numCells,1);
PenalizedMassMatPerCellTwo = zeros(size(GLB_INVP.CellMassMatrices));
for cell=1:GLB_INVP.numCells
    %%%% Primary Coefficients
    elem_mass_volume = sum( GLB_INVP.CellMassMatrices(:,:,cell)*filtered_control_at_dof(:,cell) );
    elem_penalty = elem_mass_volume / GLB_INVP.ElemVolume(cell);
    elem_penalty_1 = GLB_INVP.MinStressValue + ...
        ((1-GLB_INVP.MinStressValue) * elem_penalty^GLB_INVP.StressPower);
    von_mises_stress_square = ...
        (elem_penalty_1 * elem_avg_stress(1,cell))^2 + ...
        (elem_penalty_1 * elem_avg_stress(2,cell))^2 - ...
        ((elem_penalty_1 * elem_avg_stress(1,cell)) * ...
        (elem_penalty_1 * elem_avg_stress(2,cell))) + ...
        (3 * (elem_penalty_1 * elem_avg_stress(3,cell))^2);
    elem_von_mises_stress = sqrt(von_mises_stress_square);
    elem_stress_ratio = elem_von_mises_stress / GLB_INVP.StressNormFactor(cell);
    elem_stress_measure(cell) = GLB_INVP.ElemVolume(cell) * (elem_stress_ratio^GLB_INVP.PowerKS);
    
    elem_penalty_2 = GLB_INVP.StressPower * ...
        ((1-GLB_INVP.MinStressValue) * elem_penalty^(GLB_INVP.StressPower-1));
    elem_coeff = (2 * (elem_penalty_1 * elem_avg_stress(1,cell)) * ...
        (elem_penalty_2 * elem_avg_stress(1,cell))) + ...
        (2 * (elem_penalty_1 * elem_avg_stress(2,cell)) * ...
        (elem_penalty_2 * elem_avg_stress(2,cell))) - ...
        ( (elem_penalty_2 * elem_avg_stress(1,cell)) * ...
        (elem_penalty_1 * elem_avg_stress(2,cell)) ) - ...
        ( (elem_penalty_2 * elem_avg_stress(2,cell)) * ...
        (elem_penalty_1 * elem_avg_stress(1,cell)) ) + ...
        (6 * (elem_penalty_1 * elem_avg_stress(3,cell)) * ...
        (elem_penalty_2 * elem_avg_stress(3,cell)));
    elem_coeff = (0.5/(elem_von_mises_stress * GLB_INVP.StressNormFactor(cell))) * elem_coeff;
    elem_coeff = (GLB_INVP.constant/GLB_INVP.OriginalVolume) * ...
        GLB_INVP.ElemVolume(cell) * GLB_INVP.PowerKS * ...
        (elem_stress_ratio^(GLB_INVP.PowerKS-1)) * elem_coeff;
    
    %%%% Penalized Mass Matrix One
    PenalizedMassMatPerCellTwo(:,:,cell) = (elem_coeff / ...
        GLB_INVP.ElemVolume(cell)) * GLB_INVP.CellMassMatrices(:,:,cell);
end

%%%%%%%%%%% Assemble stress measure gradient
one = ones(GLB_INVP.nVertGrid,1);
matrix = reshape(PenalizedMassMatPerCellTwo, 1, numel(PenalizedMassMatPerCellTwo));
Mass2 = sparse(GLB_INVP.iIdxVertices, GLB_INVP.jIdxVertices, matrix);
stress_meas_grad = ((1/GLB_INVP.PowerKS) * ...
    (((GLB_INVP.constant/GLB_INVP.OriginalVolume) * ...
    sum(elem_stress_measure))^((1/GLB_INVP.PowerKS) - 1))) .* (Mass2 * one);
stress_meas_grad = GLB_INVP.Filter'*stress_meas_grad;

%%%%%%%%%%% compute perimeter control gradient
switch GLB_INVP.reg
    case{'Tikhonov'}
        perimeter_control_grad = GLB_INVP.beta * (GLB_INVP.Ms*control);
    case{'TV'}
        perimeter_control_grad = GLB_INVP.beta * 0.5 * ...
            ( 1.0 / sqrt( control' * (GLB_INVP.Ss * control) + ...
            GLB_INVP.gamma ) ) * (GLB_INVP.Ss * control);
end

output = stress_meas_grad + perimeter_control_grad;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output] = firstDerivativeWrtState(state,control)

global GLB_INVP;

%%%%%%%%%%% evaluate shear modulus at the cubature points
shear_at_dof = GLB_INVP.G( GLB_INVP.mesh.t');
shear_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(shear_at_cub_points, shear_at_dof, GLB_INVP.transformed_val_at_cub_points);
    
%%%%%%%%%%% evaluate bulk modulus at the cubature points
bulk_at_dof = GLB_INVP.B( GLB_INVP.mesh.t');
bulk_at_cub_points = zeros(GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(bulk_at_cub_points, bulk_at_dof, GLB_INVP.transformed_val_at_cub_points);

%%%%%%%%%%% evaluate strain at the cubature points
state_at_dof = state( GLB_INVP.mesh.dof );
strain_at_cub_points = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_evaluate(strain_at_cub_points, state_at_dof, GLB_INVP.Bmat);

%%%%%%%%%%% compute deviatoric strain 
dev_strain = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_tensorMultiplyDataData(dev_strain, GLB_INVP.Ddev, strain_at_cub_points);

%%%%%%%%%%% compute volumetric strain 
vol_strain = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_tensorMultiplyDataData(vol_strain, GLB_INVP.Dvol, strain_at_cub_points);

%%%%%%%%%%% compute deviatoric stress
dev_stress = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_scalarMultiplyDataData(dev_stress, shear_at_cub_points, dev_strain);

%%%%%%%%%%% compute volumetric stress
vol_stress = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numCells);
intrepid_scalarMultiplyDataData(vol_stress, bulk_at_cub_points, vol_strain);

%%%%%%%%%%% compute average stress per cell
stress_per_cell = dev_stress + vol_stress;
cell_avg_stress = zeros(GLB_INVP.numStress, GLB_INVP.numCells);
cell_avg_stress(:,:) = sum(stress_per_cell,2) ./ size(stress_per_cell,2);

%%%%%%%%%%% combine shape function gradients with shear modulus
shear_times_Bmat = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_scalarMultiplyDataField(shear_times_Bmat, shear_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, deviatoric coefficient matrix and shape function gradient
dev_stress_sensitivity = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_tensorMultiplyDataField(dev_stress_sensitivity, GLB_INVP.Ddev, shear_times_Bmat);

%%%%%%%%%%% combine shape function gradients with shear modulus
bulk_times_Bmat = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_scalarMultiplyDataField(bulk_times_Bmat, bulk_at_cub_points, GLB_INVP.Bmat);

%%%%%%%%%%% matrix-vector product, volumetric coefficient matrix and shape function gradient
vol_stress_sensitivity = zeros(GLB_INVP.numStress, GLB_INVP.numCubPoints, GLB_INVP.numDof, GLB_INVP.numCells);
intrepid_tensorMultiplyDataField(vol_stress_sensitivity, GLB_INVP.Dvol, bulk_times_Bmat);

%%%%%%%%%%% compute average stress per cell
cell_stress_sens = dev_stress_sensitivity + vol_stress_sensitivity;
cell_avg_stress_sens = zeros(GLB_INVP.numStress, GLB_INVP.numDof, GLB_INVP.numCells);
cell_avg_stress_sens(:,:,:) = sum(cell_stress_sens,2) ./ size(cell_stress_sens,2);

%%%%%%%%%%% compute objective sensitivity with respect to state
filtered_control = GLB_INVP.Filter*control;
filtered_control_at_dof = filtered_control( GLB_INVP.mesh.t');
cell_stress_measure = zeros(GLB_INVP.numCells,1);
cell_sensitivities = zeros(GLB_INVP.numDof, GLB_INVP.numCells);
for cell=1:GLB_INVP.numCells
    %%%% Primary Coefficients
    cell_mass_volume = sum( GLB_INVP.CellMassMatrices(:,:,cell)*filtered_control_at_dof(:,cell) );
    cell_penalty = cell_mass_volume / GLB_INVP.ElemVolume(cell);
    cell_penalty = GLB_INVP.MinStressValue + ...
        ((1-GLB_INVP.MinStressValue) * cell_penalty^GLB_INVP.StressPower);
    von_mises_stress_square = (cell_penalty * cell_avg_stress(1,cell))^2 + ...
        (cell_penalty * cell_avg_stress(2,cell))^2 - ...
        ((cell_penalty * cell_avg_stress(1,cell)) * ...
        (cell_penalty * cell_avg_stress(2,cell))) + ...
        (3 * (cell_penalty * cell_avg_stress(3,cell))^2);
    cell_von_mises_stress = sqrt(von_mises_stress_square);
    stress_ratio = cell_von_mises_stress / GLB_INVP.StressNormFactor(cell);
    cell_stress_measure(cell) = GLB_INVP.ElemVolume(cell) * (stress_ratio^GLB_INVP.PowerKS);
    
    cell_sensitivities(:,cell) = 2 * (cell_penalty * cell_avg_stress(1,cell)) * ...
        (cell_penalty * cell_avg_stress_sens(1,:,cell)) + ...
        2 * (cell_penalty * cell_avg_stress(2,cell)) * ...
        (cell_penalty * cell_avg_stress_sens(2,:,cell)) - ...
        ((cell_penalty * cell_avg_stress_sens(1,:,cell)) * ...
        (cell_penalty * cell_avg_stress(2,cell))) - ...
        ((cell_penalty * cell_avg_stress(1,cell)) * ...
        (cell_penalty * cell_avg_stress_sens(2,:,cell))) + ...
        6 * (cell_penalty * cell_avg_stress(3,cell)) * ...
        (cell_penalty * cell_avg_stress_sens(3,:,cell));
    cell_sensitivities(:,cell) = (GLB_INVP.constant/GLB_INVP.OriginalVolume) * ...
        (GLB_INVP.ElemVolume(cell) * GLB_INVP.PowerKS * ...
        stress_ratio^(GLB_INVP.PowerKS-1)) * (0.5/(GLB_INVP.StressNormFactor(cell) * ...
        cell_von_mises_stress)) .* cell_sensitivities(:,cell);

end
constant = ((1/GLB_INVP.PowerKS) * (((GLB_INVP.constant/GLB_INVP.OriginalVolume) * ...
    sum(cell_stress_measure))^((1/GLB_INVP.PowerKS) - 1)));
cell_sensitivities = constant .* cell_sensitivities;

%%%%%%%%%%% store force vector
reshape_sensitivities = reshape(cell_sensitivities, 1, numel(cell_sensitivities));
sparse_sensitivities = sparse(GLB_INVP.iVecIdx, GLB_INVP.iVecIdx, reshape_sensitivities);
output = spdiags(sparse_sensitivities,0);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
