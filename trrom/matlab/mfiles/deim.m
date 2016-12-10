function [P,indices] = deim(U)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Routine: Discrete Empirical Interpolation Method (DEIM)
%   Input: U = orthonormal basis
%   Output: P = binary matrixCLC

%   Output: indices = set of active indices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,index] = max(abs(U(:,1)));
P = zeros(size(U));
P(index,1) = 1;
indices = zeros(1,size(U,2));
indices(1) = index;

for i=2:size(U,2)
    PtxU = (P(:,1:i-1)')*U;
    PtxUi = (P(:,1:i-1)')*U(:,i);
    c = PtxU \ PtxUi;
    res = U(:,i) - U*c;
    [~,index] = max(abs(res));
    P(index,i) = 1;
    indices(i) = index;
end

end