clear all;close all;clc

online = load('/Users/yujia/Desktop/online1000/clip_online1000/online.txt');
online = online(:,1:6);

frame_TotalNum = max(online(:,1));
obj_TotalNum = max(online(:,2))

new_obj=[];
for obj_no = 1:obj_TotalNum
    
    frame_all_no = find(online(:,2)==obj_no);
    if isempty(frame_all_no)
        continue;
    end
    frame_start = min(frame_all_no);
    frame_end = max(frame_all_no);
    
    frame_end-frame_start;
    
    % compute overlap between first and last frames
    % remove still objects
    x = online(frame_start,3);
    y = online(frame_start,4);
    w = online(frame_start,5);
    h = online(frame_start,6);
    
    x1=max(1,(x));
    y1=max(1,(y));
    x2=min(640,(x)+(w));
    y2=min(360,(y)+(h));
    
    x = online(frame_end,3);
    y = online(frame_end,4);
    w_ = online(frame_end,5);
    h_ = online(frame_end,6);
    x1_=max(1,(x));
    y1_=max(1,(y));
    x2_=min(640,(x)+(w_));
    y2_=min(360,(y)+(h_));
    
    A = (min(x2,x2_)-max(x1,x1_))*(min(y2,y2_)-max(y1,y1_));
    overlap = A/(w*h+w_*h_-A);
    if (min(x2,x2_)-max(x1,x1_))<0
        overlap=0;
    elseif (min(y2,y2_)-max(y1,y1_))<0
        overlap=0;
    end
    
    if overlap > 0.8
        
        continue;
    end
    
    new_obj=[new_obj obj_no];
    
    first_width = online(frame_start,5);
    first_height = online(frame_start,6);
    last_width = online(frame_end,5);
    last_height = online(frame_end,6);
    avg_width = (first_width + last_width)/2;
    avg_height = (first_height + last_width)/2;
    
    clip = {};
    
    
    frame_tmp = 1;
    for frame_no = frame_start:frame_end   %%%frame_no --> obj_index
        
        img_path = ['/Users/yujia/Desktop/online1000/img/img_',...
            num2str(online(frame_no,1),'%06d'),'.jpg'];
        im = imread(img_path);
        
        x = online(frame_no,3);
        y = online(frame_no,4);
        w = online(frame_no,5);
        h = online(frame_no,6);
        
        x1=max(1,(x)-(w/2));
        y1=max(1,(y)-(h/2));
        x2=min(size(im,2),(x)+(2*w));
        y2=min(size(im,1),(y)+(2*h));
        
        im_crop = im(y1:y2,x1:x2,:);
        im_crop_resize = imresize(im_crop,[2*avg_height,2*avg_width]);
        %imshow(im_crop_resize);
        clip{frame_tmp,1} = im_crop_resize;
        frame_tmp = frame_tmp + 1;
        
    end
    
    clip_new{1,1} = frame_start;
    clip_new{2,1} = frame_end;
    clip_new{3,1} = clip;
    
    clip_filename = ['/Users/yujia/Desktop/clip_', num2str(obj_no)];
    save(clip_filename,'clip_new');
    
end