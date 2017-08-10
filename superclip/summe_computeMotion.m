function [ motion_magnitude,motion_magnitude_back ] = summe_computeMotion(imageList,frameRange,FPS,Params, clip)
%summe_computeMotion Computes the motion magnitude over a range of frames   

    fprintf('Compute forward motion\n');    
%     frames=imageList(frameRange(1):frameRange(2));
    
%     clip = load('clip_6');
    
    motion_magnitude=getMagnitude(clip,Params,FPS);
    
    fprintf('Compute backward motion\n')    
%     frames=imageList(frameRange(2):-1:frameRange(1));
    clip_ = flipud(clip.clip_new{3});
    
    motion_magnitude_back=getMagnitude(clip,Params,FPS);
    motion_magnitude_back=flip(motion_magnitude_back);
    
end
    
function [motion_magnitude]=getMagnitude(clip,Params,FPS)
    clip_start = clip.clip_new{1};
    clip_end = clip.clip_new{2};
    clips = clip.clip_new{3};
    
    motion_magnitude=zeros(clip_end-clip_start+1,1);
    for startFrame=1:Params.stepSize:clip_end-clip_start+1-Params.stepSize
        % Load the first image
        frame = clips{startFrame};
        

        if ~exist('frameSize','var')
            frameSize=sqrt(size(frame,1)*size(frame,2));
        end
        if ~exist('new_points','var')
            old_points=zeros(0,2);
        else
            old_points=new_points(points_validity,:);
        end


        % Detect points
        minQual=Params.minQual;
        points=[];
        tries=0;
        while (size(old_points,1)+size(points,1)) < Params.num_tracks*0.95 && tries<5 % we reinitialize only, if we have too little points
            points=detectFASTFeatures(rgb2gray(frame),'MinQuality',minQual);
            minQual=minQual/5;
            tries=tries+1;
        end        
        if numel(points) > 0
            old_points=[old_points; points.Location];
        end
    
        if size(old_points,1) > Params.num_tracks
            indices=randperm(size(old_points,1));
            old_points=old_points(indices(1:Params.num_tracks),:);
        end


        % Compute magnitude
        if (length(old_points) >= Params.min_tracks) % if at least k points are detected
            % Initialize tracker
            pointTracker = vision.PointTracker;
            initialize(pointTracker,old_points,frame);
            for frameNr=1:Params.stepSize-1
                frame = clips{startFrame+frameNr};
                [new_points,points_validity] = step(pointTracker,frame);
            end

            diff=new_points(points_validity,:)-old_points(points_validity,:);
            diff=mean(norm(diff));

            % add it to the array and normalize by frame size
            motion_magnitude(startFrame:startFrame+Params.stepSize-1)=(FPS/Params.stepSize)*diff./frameSize;
        end
    end
end

