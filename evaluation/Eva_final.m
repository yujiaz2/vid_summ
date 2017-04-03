clear all;close all;clc

frame_cur_end = 0;
% loss_res = load('/Users/yujia/Desktop/loss_res_sum.mat');
% loss_res = loss_res.sum;
loss_res = importdata('/Users/yujia/Desktop/loss_res_3.txt');
% loss_res = load('/Users/yujia/Desktop/sum.mat');
% loss_res = loss_res.sum;
frames_no = importdata('feat_9parts.txt');

DDsum = load('Dsum.mat');
precision = [];
recall = [];
tpr = [];
fpr = [];
f = [];

% recon_error = load('recon_error.mat');

for sample_no = 1:30
    a= loadjson(['vid' num2str(sample_no) '.json']);
    a=struct2cell(a);
    
    for i=1:length(a)
        tmp=a{i};
        GTsum(i,1)=tmp(1,1);
        GTsum(i,2)=tmp(end,1);
        GTsum(i,3:6)=tmp(1,2:5);
    end

frame_cur_start = frame_cur_end + 1;
frame_cur_end = frame_cur_end + frames_no(sample_no*9,1);

loss_cur = loss_res(frame_cur_start:frame_cur_end,1);
% loss_cur = recon_error.recon_error{sample_no,1};

Dsum = [];
Dsum = DDsum.DDsum{1,sample_no};

Dsum = [Dsum loss_cur];

Dsum(:,5)=Dsum(:,5)+Dsum(:,3);
Dsum(:,6)=Dsum(:,6)+Dsum(:,4);
Dsum = [Dsum loss_cur];

numDsum=size(Dsum,1);
numGT=size(GTsum,1);
GT_tempor_area=GTsum(:,2)-GTsum(:,1);
GT_spatial_area=(GTsum(:,5)-GTsum(:,3)).*(GTsum(:,6)-GTsum(:,4));
threshold=0.1;

D_gt=zeros(numDsum,1);
D_temporal=zeros(numDsum,1);
D_spatial=zeros(numDsum,1);
for i=1:numDsum
    tmpsum=repmat(Dsum(i,:),[numGT,1]);
    tmp_temporal_start=max(tmpsum(:,1),GTsum(:,1));
    tmp_temporal_end=min(tmpsum(:,2),GTsum(:,2));
    tmp_temporal_overlap=(tmp_temporal_end-tmp_temporal_start)./(tmpsum(:,2)-tmpsum(:,1)+GT_tempor_area-tmp_temporal_end+tmp_temporal_start);
    D_temporal(i)=max(tmp_temporal_overlap);
    tmp_temporal_overlap=tmp_temporal_overlap>threshold;
    
    
    tmp_spatial_start_x=max(tmpsum(:,3),GTsum(:,3));
    tmp_spatial_end_x=min(tmpsum(:,5),GTsum(:,5));
    a=tmp_spatial_end_x-tmp_spatial_start_x;
    a(a<0)=0;
    if max(a)==0
        spatial_overlap(i)=0;
        continue;
    end
    tmp_spatial_start_y=max(tmpsum(:,4),GTsum(:,4));
    tmp_spatial_end_y=min(tmpsum(:,6),GTsum(:,6));
    b=tmp_spatial_end_y-tmp_spatial_start_y;
    b(b<0)=0;
    if max(b)==0
        spatial_overlap(i)=0;
        continue;
    end
    tmp_spatial_top=a.*b;
    tmp_spatial_down=GT_spatial_area+(tmpsum(:,5)-tmpsum(:,3)).*(tmpsum(:,6)-tmpsum(:,4))-tmp_spatial_top;
    D_spatial(i)=max(tmp_spatial_top./tmp_spatial_down);
    
    tmp_spatial_overlap=(tmp_spatial_top./tmp_spatial_down)>threshold;
    tmp_overlap=tmp_temporal_overlap.*tmp_spatial_overlap;
    D_gt(i)=max(tmp_overlap);
end

[Precision,Recall,TPR, FPR, AUC,AP,F] = QXL_ROC( Dsum(:,end), D_gt, 100 );
precision = [precision; Precision];
tpr = [tpr; TPR];
fpr = [fpr; FPR];

f = [f;F];

end

Pre = mean(precision);
Tpr = mean(tpr);
Fpr = mean(fpr);
F_ = mean(f);

% F(i+1)=TP*Precision/(TP+1*Precision)*2;

%Result.F=Tpr(50)*Pre(50)/(Tpr(50)+1*Pre(50))*2;
Result.F=F_(50);
Result.AP=-trapz(Tpr,Pre);
Result.AUC=-trapz(Fpr,Tpr);

% plot(TPR,FPR,'b');
% hold on
% c=polyfit(TPR,FPR,3);
% xi=linspace(0,1,100);
% yi=polyval(c,xi);
% plot(TPR,FPR,'o',xi,yi);
% hold on
% plot(Precision,Recall,'g');
% hold on
% d=polyfit(Precision,Recall,3);
% xj=linspace(0,1,100);
% yj=polyval(d,xj);
% plot(Precision,Recall,'r',xj,yj);

%num2str(k1,'%03d')