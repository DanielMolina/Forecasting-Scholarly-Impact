% linear  Regression
clear;

load('outputt.mat');
%colomn 1 is index, colomn 3 is citation count(output) the rest are
%features
feat = [ones(size(output,1),1),output(:,2),output(:,4:end)];   % We add a colomns of ones to allow for constants in the model
cit = output(:,3);

% Adding features created from multiplying other features (Polynomial of degree 2) 
for i=2:4
    for j= i:4
      feat = [feat, feat(:,i).*feat(:,j)];    
    end
end
    
n = 2000000;                   %train size
n_t = size(output,1)-2000000;  %test size
seed = 1;  % in order to compare different models we should have same train/test sets. 
rng(seed);

for i=1:10
    test_ind = randi([1 size(cit,1)],n_t,1);      % randomly choose n_t indices for test
    train_ind = setdiff(1:size(cit,1),test_ind);  % the rest of indices are used for training
    % create train and test sets
    cit_t = cit(test_ind);
    feat_t = feat(test_ind,:);
    cit_r = cit(train_ind);
    feat_r = feat(train_ind,:);
    % calculate the optimal weight vector
    b = regress(cit_r,feat_r); 
    % predict the citation c
    pred_cit = feat_t*b;
    % calculate the error and MAE and RMSE
    err(:,i) = cit_t-pred_cit;
    MAE(i) = mean(abs(err(:,i))); 
    RMSE(i) = (mean(err(:,i).^2))^.5; 
end
% take the average over 10 runs
MMAE = mean(MAE); 
MRMSE = mean(RMSE); 
save('LR','MMAE','MRMSE','err')

% plot(1:n_t,cit_t,'.')
% hold on; 
% plot(1:n_t,pred_cit,'.r')
% xlabel('data index')
% ylabel('citation')

