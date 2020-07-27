function [matching_result, objective_score]=CompareRealGraphs(A,B)
% 1. Integer Projected Fixed Point Method (IPFP)
% 2. Spectral Matching with Affine Constraints (SMAC)
% 3. Probabilistic Graph Matching (PM)
% 4. Reweighted Random Walk Matching (RRWM)
% 5. Factorized Graph Matching (FGM)
% 6. Kernelized Graph Matching (KerGM)

%algorithm parameter

% % IPFP-U
% asgIpfpU = gm(K, Ct, asgT, pars{5}{:});
% AccTotal{1}.res(irep,jvar)=asgIpfpU.acc;
% 
% % SMAC
% asgSmac = gm(K, Ct, asgT, pars{4}{:});
% AccTotal{2}.res(irep,jvar)=asgSmac.acc;
% 
% % PM
% asgPm = pm(K, KQ, gphs, asgT);
% AccTotal{3}.res(irep,jvar)=asgPm.acc;
% 
% % RRWM
% asgRrwm = gm(K, Ct, asgT, pars{7}{:});
% AccTotal{4}.res(irep,jvar)=asgRrwm.acc;
% 
% % FGM-U
% asgFgmU = fgmU(KP, KQ, Ct, gphs, asgT, pars{8}{:});
% AccTotal{5}.res(irep,jvar)=asgFgmU.acc;

% KerGM
lambda=0.005; num=11;para.gamma=5; para.D=20;
[matching_result,objective_score]=KerGM_Pathfollowing_RandFourierFeature(A,B,lambda,num,para);
% matching_result = X;
% acc = matchAsg(X, asgT);
% AccTotal{6}.res(irep,jvar)=acc;

end