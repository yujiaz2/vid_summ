clear all;close all;clc

for i=1:30
    error = [];
    feat_9parts = importdata('/Users/yujia/Desktop/feat_9parts.txt');
    feat_cur = importdata(['/Users/yujia/Desktop/online_all_30samples/1-30/5. combined_feature!!!/feat_online' num2str(i, '%02d') '.txt']);
    len = size(feat_cur,1);
    avg = (feat_cur(1:3:len,:)+feat_cur(2:3:len,:)+feat_cur(3:3:len,:))/3;
    [COEFF,SCORE] = princomp(avg);
    input = SCORE(:,1:256);
    input = input';
        
    feat_rows = feat_9parts((i-1)*9+1:i*9,:);
    
    param.K=size(avg,1)*0.5;  % learns a dictionary with 100 elements
    param.lambda=0.15;
    param.numThreads=-1; % number of threads
    param.batchsize=400;
    param.verbose=false;
    param.iter=100;
    
    for j=1:9
        X = input(:,1:feat_rows(j));
        D = mexTrainDL(X,param);
        param.mode=2;
        alpha=mexLasso(X,D,param);
    
        recon_err = 0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha));
%         err_normalize = mapminmax(recon_err,0,1);
        
        if j==1
            error(1,1:feat_rows(j))=recon_err;
        else
            error(1,feat_rows(j-1)+1:feat_rows(j))=recon_err(1,feat_rows(j-1)+1:feat_rows(j));
        end
    end
    
    filenm = ['error_'    num2str(i,'%02d') '.mat'];
    save(filenm,'error', '-mat');

end