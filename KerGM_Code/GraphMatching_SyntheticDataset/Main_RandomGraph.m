% In this file, we compare different state-of-the-art Graph Matching
% algorihtms, including:
% 1. Integer Projected Fixed Point Method (IPFP)
% 2. Spectral Matching with Affine Constraints (SMAC)
% 3. Probabilistic Graph Matching (PM)
% 4. Reweighted Random Walk Matching (RRWM)
% 5. Factorized Graph Matching (FGM)
% 6. Kernelized Graph Matching (KerGM)

setting=5;

nrep=100; % We repeat the experiment nrep times.
switch setting
    case 0
        %load the existing graphs
        AccTotal_out=InitializeAccTotal(1,1);
        a = load("C:\Tomer\courses\deep learning\data\Baboon1_FinalFinal2.mat");
        graph = a.conmatT(:,:,1);
        [A1,A2,GT,gphs,KQ,K]=GenerateSpecificGraph(graph);
        [AccTotal_out, objective_score]=CompareAlgorithms(A1,A2,GT,gphs,KQ,K,AccTotal_out,...
            1, 1);
        fprintf('done acc %dth ', AccTotal_out)
    case 6
        %load the existing graphs
        AccTotal_out=InitializeAccTotal(1,1);
        graph_try=rand(5);
        graph_try=(graph_try+graph_try')/2; 
        graph_try=graph_try-diag(diag(graph_try));
        [A1,A2,GT,gphs,KQ,K]=GenerateSpecificGraph(graph_try);
        [AccTotal_out, objective_score]=CompareAlgorithms(A1,A2,GT,gphs,KQ,K,AccTotal_out,...
            1, 1);
        fprintf('done acc %dth ', AccTotal_out)
    case 5
        %load the existing graphs
        AccTotal_out=InitializeAccTotal(1,1);
        baboon_1 = load("C:\Tomer\courses\deep learning\data\xFinalFinal2\Baboon1_FinalFinal2.mat");
        baboon_1 = baboon_1.conmatT(:,:,1);
        baboon_2 = load("C:\Tomer\courses\deep learning\data\xFinalFinal2\Baboon2_FinalFinal2.mat");
        baboon_2 = baboon_2.conmatT(:,:,1);
        orangutang_1 = load("C:\Tomer\courses\deep learning\data\xFinalFinal2\Ferret1_FinalFinal2.mat");
        orangutang_1 = orangutang_1.conmatT(:,:,1);
        orangutang_2 = load("C:\Tomer\courses\deep learning\data\xFinalFinal2\Ferret2_FinalFinal2.mat");
        orangutang_2 = orangutang_2.conmatT(:,:,1);
        macaque_1 = load("C:\Tomer\courses\deep learning\data\xFinalFinal2\Macaque1_FinalFinal2.mat");
        macaque_1 = macaque_1.conmatT(:,:,1);
        rat_1 = load("C:\Tomer\courses\deep learning\data\xFinalFinal2\Rat1_FinalFinal2.mat");
        rat_1 = rat_1.conmatT(:,:,1);
        [matching_orangutang_1_223, objective_score_same23]=CompareRealGraphs(orangutang_1,rat_1);
        [matching_orangutang_1_224, objective_score_same24]=CompareRealGraphs(rat_1,macaque_1);
        [matching_orangutang_1_22, objective_score_same2]=CompareRealGraphs(orangutang_1,macaque_1);
        [matching_orangutang_1_2, objective_score_same]=CompareRealGraphs(orangutang_1,orangutang_2);
        [matching_baboon_1_orangutang_2, objective_score_different]=CompareRealGraphs(baboon_1,orangutang_2);
        [matching_baboon_1_orangutang_1, objective_score_different2]=CompareRealGraphs(baboon_1,orangutang_1);
        [matching_baboon_1_1, objective_score_different3]=CompareRealGraphs(baboon_1,baboon_1);
        [matching_baboon_1_2, objective_score_different4]=CompareRealGraphs(baboon_1,baboon_2);
        save('C:\Tomer\courses\deep learning\data\pairwise_matching\orangutang_1_and_2_constraint.mat', 'matching_same')
        save('C:\Tomer\courses\deep learning\data\pairwise_matching\baboon_1_and_2_constraint.mat', 'matching_different')
        fprintf('done acc %dth ', AccTotal_out)
    case 4
        inlier=4; outlier=2; density=1; deformation=0;
        nvar=length(outlier); Acc=zeros(nrep,nvar);number=0;
        AccTotal_out=InitializeAccTotal(nrep,nvar);             
        % Generate random graphs
        [A1,A2,GT,gphs,KQ,K]=GenerateRandGraph(inlier,outlier,...
            density,deformation);
        % Compare different algorithms
         [AccTotal_out, objective_score]=CompareAlgorithms(A1,A2,GT,gphs,KQ,K,AccTotal_out,...
             1, 1);
        number=number+1;
        fprintf('The %dth graph matching problem has been solved\n', number);
    case 1
        inlier=50; outlier=0:5:50; density=1; deformation=0;
        nvar=length(outlier); Acc=zeros(nrep,nvar);number=0;
        AccTotal_out=InitializeAccTotal(nrep,nvar);
        for irep=1:nrep
            for jvar=1:nvar               
                % Generate random graphs
                [A1,A2,GT,gphs,KQ,K]=GenerateRandGraph(inlier,outlier(jvar),...
                    density,deformation);
                % Compare different algorithms
                 [AccTotal_out]=CompareAlgorithms(A1,A2,GT,gphs,KQ,K,AccTotal_out,...
                     irep, jvar);
                number=number+1;
                fprintf('The %dth graph matching problem has been solved\n', number);                
            end
        end
        save('outlier.mat', 'AccTotal_out');        
    case 2
        inlier=50; outlier=5; density=0.3:0.1:1; deformation=0.1;
        nvar=length(density); Acc=zeros(nrep,nvar); number=0;
        AccTotal_den=InitializeAccTotal(nrep,nvar);
        for irep=1:nrep
           for jvar=1:nvar               
               % Generate random graphs
               [A,B,GT,gphs,KQ,K]=GenerateRandGraph(inlier,outlier,...
                    density(jvar),deformation);
                % Compare different algorithms
                 [AccTotal_den]=CompareAlgorithms(A,B,GT,gphs,KQ,K,AccTotal_den,...
                     irep, jvar);
                number=number+1;
                fprintf('The %dth graph matching problem has been solved\n', number);                
            end
        end
        save('densitynew_outlier5noise01.mat','AccTotal_den');
    case 3 
        inlier=50; outlier=0; density=1; deformation=0:0.02:0.2;
        nvar=length(deformation); Acc=zeros(nrep,nvar);number=0;
        AccTotal_noise=InitializeAccTotal(nrep,nvar);
        for irep=1:nrep
           for jvar=1:nvar               
               % Generate random graphs
               [A,B,GT,gphs,KQ,K]=GenerateRandGraph(inlier,outlier,...
                    density,deformation(jvar));
                % Compare different algorithms
                 [AccTotal_noise]=CompareAlgorithms(A,B,GT,gphs,KQ,K,AccTotal_noise,...
                     irep, jvar);
                number=number+1;
                fprintf('The %dth graph matching problem has been solved\n', number);                
            end
        end
        save('noise.mat','AccTotal_noise');
end 


