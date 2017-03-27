clear all;close all;clc

online = load('online.txt');
online = online(:,1:6);
object_clip = online;
object_clip(:,[1,2])=object_clip(:,[2,1]);

superframe_num = max(object_clip(:,1));
for i = 1:superframe_num
    if exist(['/Users/yujia/Desktop/online_test/superframe/superframe_' num2str(i,'%d') '.mat'],'file')
        superframe = load(['/Users/yujia/Desktop/online_test/superframe/superframe_' num2str(i,'%d')]);
        sf_length = size(superframe.superFrames,1);
        sf_index = ones(superframe.superFrames(sf_length,2),1);
        for j = 1:sf_length
            sf_index(superframe.superFrames(j,1):superframe.superFrames(j,2),:) = j;
        end
        object_clip(object_clip(:,1)==i,7) = sf_index;
    else
        object_clip(object_clip(:,1)==i,:)=[];
    end
end

frame_TotalNum = max(online(:,1));
obj_TotalNum = max(online(:,2));

save object_clip object_clip