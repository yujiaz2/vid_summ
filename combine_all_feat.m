clear all;close all;clc

a=[];
% maxj=3898;
maxj=1000;


for iteration = 1:30
    path = ['/Users/yujia/Desktop/online_all_30samples/1-30/4. clipFeat/clipFeat_online', num2str(iteration,'%02d')];
    
    a=[];
    cnt=1;
    for j=1:maxj
        i=1;
        while exist([path, '/clipFeat_obj_', num2str(j,'%03d'), '_sf_', num2str(i,'%03d'), '.txt'],'file')
            filename = [path, '/clipFeat_obj_', num2str(j,'%03d'), '_sf_', num2str(i,'%03d'), '.txt'];
            txt=importdata(filename);
            a=[a;txt];
            [j i]
            i=i+1;
        end
    end
    
    feat = a;
    
    % save feat_all.mat feat
    file_name = ['feat_online',num2str(iteration,'%02d'),'.txt'];
    save(file_name,'a','-ASCII')
    
end