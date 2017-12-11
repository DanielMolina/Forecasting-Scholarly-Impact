clear
load('outputt.mat');
%colomn 1 is index, colomn 3 is citation count(output) the rest are
%features
feat = [output(:,2),output(:,4:end)];
cit = output(:,3);
n = 2000000;                   %train size
n_t = size(output,1)-2000000;  %test size
seed = 1;  % in order to compare different models we should have same train/test sets. 
rng(seed);
for i=1:10
    test_ind = randi([1 size(cit,1)],n_t,1);      % randomly choose n_t indices for test
    train_ind = setdiff(1:size(cit,1),test_ind);  % the rest of indices are used for training
    % create train and test sets
    cit_t = output(test_ind,3);
    feat_t = feat(test_ind,:);
    cit_r = output(train_ind,3);
    feat_r = feat(train_ind,:);
    % Create model based on kNN with 10 neighbors
    Mdl = fitcknn(feat_r,cit_r);  
    Mdl.NumNeighbors = 10;
    % predict labels using the model
    pr_cit = predict(Mdl,feat_t);
    % calculate the error and MAE and RMSE
    err(:,i) = cit_t-pr_cit;
    MAE(i) = mean(abs(err(:,i))); 
    RMSE(i) = (mean(err(:,i).^2))^.5; 
end
% take the average over 10 runs
MMAE = mean(MAE); 
MRMSE = mean(RMSE); 
save('Knn','MMAE','MRMSE','err')

% plot(1:n_t,cit_t,'.')
% hold on; plot(1:n_t,pr_cit,'.r')
% xlabel('data index')
% ylabel('citation')

