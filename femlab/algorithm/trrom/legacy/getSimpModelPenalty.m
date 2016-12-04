function [output] = getSimpModelPenalty(control, derivative_t)

global GLB_INVP;

power = GLB_INVP.SimpPenalty;
control_at_dof = control( GLB_INVP.mesh.t');
average_control = zeros(GLB_INVP.numCells,1);
for cell=1:GLB_INVP.numCells
    average_control(cell) = ...
        sum(GLB_INVP.CellMassMatrices(:,:,cell)*control_at_dof(:,cell)) ...
        / GLB_INVP.ElemVolume(cell);
end

switch derivative_t
    case 'zero'
        output = average_control.^power;
    case 'first'
        output = power .* (average_control.^(power-1));
    case 'second'
        penalty = power * (power-1);
        output = penalty .* (average_control.^(power-2));
end

end
