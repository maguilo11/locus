function [weights] = Filter(mesh,avg_cell_weights,radius)

weights = zeros(size(mesh.p,1));
for i=1:size(mesh.p,1)
    factor = 0;
    for j=1:size(mesh.p,1)
        d = sqrt((mesh.p(i,1) - mesh.p(j,1))^2 + (mesh.p(i,2) - mesh.p(j,2))^2);
        if(d < radius)
            weights(i,j) = (radius - d)*avg_cell_weights;
            factor = factor + weights(i,j);
        end
    end
    weights(i,:) = (1/factor) .* weights(i,:);
end

end