%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Superframe segmentation
%%%%%%%%
% This shows how to use superframes as a temporal segmentation. 
% The code is a modified version of the one used in the initial paper
%%%%%%%%
% publication: Gygli et al. - Creating Summaries from User Videos, ECCV 2014
% author:      Michael Gygli, PhD student, ETH Zurich,
% mail:        gygli@vision.ee.ethz.ch
%%%%%%%%

%%%%% Video parameters  %%%%%
HOMEIMAGES='./example_frames'; % Directory containing video frames
FPS=30; % Needed to use be able to compute the optimal segment length

%%%%% Method parameters %%%%%
% Default
default_parameters;

% Length prior parameters (optional, otherwise we take the ones learnt)
% Use can use this to change the length of the superframes
% See paper for more information
Params.lognormal.mu=1.16571;
Params.lognormal.sigma=0.742374;

% Read in the images
images=dir(fullfile(HOMEIMAGES,'*.jpg'));
imageList=cellfun(@(X)(fullfile(HOMEIMAGES,X)),{images(:).name},'UniformOutput',false);

%% Run Superframe segmentation

clip_total_num = 307;
for i=1:clip_total_num
    if exist(['/Users/yujia/Desktop/online_test/clip/clip_' num2str(i,'%d') '.mat'],'file')
        [superFrames,motion_magnitude_of] = summe_superframeSegmentation(imageList,FPS,Params, i);
        save(['superframe_' num2str(i,'%d') '.mat'], 'superFrames');
    end
end