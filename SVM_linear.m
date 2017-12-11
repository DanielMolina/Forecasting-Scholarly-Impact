clear
load('outputt.mat');
feat = [output(:,2),output(:,4),output(:,6:end)];  
cit = output(:,3);

n = 2000000;
n_t = size(output,1)-2000000;
seed = 1;
rng(seed);
for i=1:10
    test_ind = randi([1 size(cit,1)],n_t,1);      % randomly choose n_t indices for test
    train_ind = setdiff(1:size(cit,1),test_ind);  % the rest of indices are used for training
    % create train and test sets
    cit_t = output(test_ind,3);
    feat_t = feat(test_ind,:);
    cit_r = output(train_ind,3);
    feat_r = feat(train_ind,:);
    Mdl = fitrsvm(feat_r,cit_r);
    pred_cit = predict(Mdl,feat_t);
    err(:,i) = cit_t-pred_cit;
    MAE(i) = mean(abs(err(:,i))); 
    RMSE(i) = (mean(err(:,i).^2))^.5;
end
MMAE = mean(MAE); 
MRMSE = mean(RMSE); 
save('LSVM','MMAE','MRMSE','err')

% plot(1:n_t,cit_t,'.')
% hold on; plot(1:n_t,pr_cit,'.r')
% xlabel('data index')
% ylabel('citation')
