function [basis] = pod(threshold,singular_values,singular_vectors,snapshots)

% Gather maximum energy eigenvectors
eigenVectorCount = 1;
eigenvalues = diag(singular_values);
totalEnergy = sum(eigenvalues);
while(true)
    energy = sum(eigenvalues(1:eigenVectorCount))/totalEnergy;
    if(energy >= threshold)
        break;
    end
    eigenVectorCount = eigenVectorCount + 1;
end
% Compute new orthogonal basis
values = 1 ./ sqrt(eigenvalues);
basis = snapshots(:,1:end)*singular_vectors(:,1:eigenVectorCount);
for i=1:eigenVectorCount
    basis(:,i) = values(i) .* basis(:,i);
end

end