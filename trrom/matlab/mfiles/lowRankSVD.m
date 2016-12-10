function [SVal,LSVec,RSVec] = ...
    lowRankSVD(new_snapshots, old_SingularVal, old_LeftSingularVec, old_RightSingularVec)
% low rank singular value decomposition
% Author: Miguel A. Aguilo
M = old_LeftSingularVec'*new_snapshots;
P = new_snapshots - old_LeftSingularVec*M;
[Q,R]=qr(P,0);
zero = zeros(size(R,1),size(old_SingularVal,2));
K = [old_SingularVal, M; ...
     zero, R];
[C,SVal,D] = svd(K,'econ');
LSVec = [old_LeftSingularVec Q]*C;
RSVec = [old_RightSingularVec zeros(2); zeros(2) eye(2)]*D;
end