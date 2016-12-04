function [P] = deim(U)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Routine: Discrete Empirical Interpolation Method (DEIM)
%   Input: U = orthonormal basis
%   Output: P = index matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = zeros(size(U));
[~,index] = max(abs(U(:,1)));
P(index,1) = 1;
for i=2:size(U,2)
    PtxU = (P(:,1:i-1)')*U;
    PtxUi = (P(:,1:i-1)')*U(:,i);
    c = PtxU \ PtxUi;
    res = U(:,i) - U*c;
    [~,index] = max(abs(res));
    P(index,i) = 1;
end

end