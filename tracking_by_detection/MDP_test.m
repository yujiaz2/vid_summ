% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
%
% testing MDP
function metrics = MDP_test(seq_idx, seq_set, tracker)
%image_show = 0;

is_show = 1;   % set is_show to 1 to show tracking results in testing
is_save = 1;   % set is_save to 1 to save tracking result
is_text = 0;   % set is_text to 1 to display detailed info
is_pause = 0;  % set is_pause to 1 to debug

opt = globals();
opt.is_text = is_text;
opt.exit_threshold = 0.7;

if is_show
    close all;
end

if strcmp(seq_set, 'train') == 1
    seq_name = opt.mot2d_train_seqs{seq_idx};
    seq_num = opt.mot2d_train_nums(seq_idx);
else
    seq_name = opt.mot2d_test_seqs{seq_idx};
    seq_num = opt.mot2d_test_nums(seq_idx);
end

txtpath='/Users/yujia/Desktop/frame_level/bbox/1/';
txts=dir([txtpath '*.txt']);
frame_num = length(txts);

dres_image = {};
dres_image.x = ones(frame_num,1);
dres_image.y = ones(frame_num,1);
dres_image.w = 640*ones(frame_num,1);
dres_image.h = 360*ones(frame_num,1);


for frame = 1:frame_num
    dres_image.I{frame,1} = imread(['/Users/yujia/Desktop/frame_level/img/1/' num2str(frame,'%06d') '.jpg']);
    dres_image.Igray{frame,1} = rgb2gray(dres_image.I{frame,1});
end

dres_det = {};
dres_det.fr = ones(frame_num,1);
dres_det.id = -1*ones(frame_num,1);
dres_det.x = ones(frame_num,1);
dres_det.y = ones(frame_num,1);
dres_det.w = ones(frame_num,1);
dres_det.h = ones(frame_num,1);
dres_det.r = ones(frame_num,1);

index=1;
for frame_no = 1:frame_num
    bbox_path = ['/Users/yujia/Desktop/frame_level/bbox/1/bbox_' num2str(frame_no,'%06d') '.txt'];
    if exist(bbox_path,'file')==0
        continue;
    end
    
    bbox = load(bbox_path);
    score_path = ['/Users/yujia/Desktop/frame_level/score/1/score_' num2str(frame_no,'%06d') '.txt'];
    score = load(score_path);
    
    for obj_no = 1:size(bbox,1)/4;
        dres_det.fr(index) = frame_no;
        dres_det.x(index)=bbox(obj_no*4-3);
        dres_det.y(index)=bbox(obj_no*4-2);
        dres_det.w(index)=bbox(obj_no*4-1)-bbox(obj_no*4-3);
        dres_det.h(index)=bbox(obj_no*4)-bbox(obj_no*4-2);
        dres_det.r(index)=score(obj_no)*100;
        index = index+1;
    end
end
dres_det.id = -1*ones(size(dres_det.fr,1),1);

if strcmp(seq_set, 'train') == 1
    % read ground truth
    filename = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'gt', 'gt.txt');
    dres_gt = read_mot2dres(filename);
    dres_gt = fix_groundtruth(seq_name, dres_gt);
end

% load the trained model
if nargin < 3
    object = load('tracker.mat');
    tracker = object.tracker;
end

% intialize tracker
I = dres_image.I{1};
tracker = MDP_initialize_test(tracker, size(I,2), size(I,1), dres_det, is_show);

% for each frame
trackers = [];
id = 0;
for fr = 1:frame_num %seq_num
    fr
    
    if is_text
        
    else
        fprintf('.');
        if mod(fr, 100) == 0
            
        end
    end
    
    % extract detection
    index = find(dres_det.fr == fr);
    dres = sub(dres_det, index);
    
    dres = MDP_crop_image_box(dres, dres_image.Igray{fr}, tracker);
    
    % sort trackers
    index_track = sort_trackers(trackers);
    
    % process trackers
    for i = 1:numel(index_track)
        ind = index_track(i);
        
        if trackers{ind}.state == 2
            % track target
            trackers{ind} = track(fr, dres_image, dres, trackers{ind}, opt);
            % connect target
            if trackers{ind}.state == 3
                [dres_tmp, index] = generate_initial_index(trackers(index_track(1:i-1)), dres);
                dres_associate = sub(dres_tmp, index);
                trackers{ind} = associate(fr, dres_image,  dres_associate, trackers{ind}, opt);
            end
        elseif trackers{ind}.state == 3
            % associate target
            [dres_tmp, index] = generate_initial_index(trackers(index_track(1:i-1)), dres);
            dres_associate = sub(dres_tmp, index);
            trackers{ind} = associate(fr, dres_image, dres_associate, trackers{ind}, opt);
        end
    end
    
    % find detections for initialization
    [dres, index] = generate_initial_index(trackers, dres);
    for i = 1:numel(index)
        % extract features
        dres_one = sub(dres, index(i));
        f = MDP_feature_active(tracker, dres_one);
        % prediction
        label = svmpredict(1, f, tracker.w_active, '-q');
        % make a decision
        if label < 0
            continue;
        end
        
        % reset tracker
        tracker.prev_state = 1;
        tracker.state = 1;
        id = id + 1;
        
        trackers{end+1} = initialize(fr, dres_image, id, dres, index(i), tracker);
    end
    
    % resolve tracker conflict
    trackers = resolve(trackers, dres, opt);
    
    dres_track = generate_results(trackers);
    %
    figure(1);
    
    show_dres(fr, dres_image.I{fr}, 'Tracking', dres_track, 2);
    
    pause(0.01);
    
end

% write tracking results
filename = sprintf('%s/online.txt', opt.results);
fprintf('write results: %s\n', filename);
write_tracking_results(filename, dres_track, opt.tracked);

% evaluation
if strcmp(seq_set, 'train') == 1
    benchmark_dir = fullfile(opt.mot, opt.mot2d, seq_set, filesep);
    metrics = evaluateTracking({seq_name}, opt.results, benchmark_dir);
else
    metrics = [];
end

% save results
if is_save
    filename = sprintf('%s/online_results.mat', opt.results);
    save(filename, 'dres_track', 'metrics');
end


% sort trackers according to number of tracked frames
function index = sort_trackers(trackers)

sep = 10;
num = numel(trackers);
len = zeros(num, 1);
state = zeros(num, 1);
for i = 1:num
    len(i) = trackers{i}.streak_tracked;
    state(i) = trackers{i}.state;
end

index1 = find(len > sep);
[~, ind] = sort(state(index1));
index1 = index1(ind);

index2 = find(len <= sep);
[~, ind] = sort(state(index2));
index2 = index2(ind);
index = [index1; index2];

% initialize a tracker
% dres: detections
function tracker = initialize(fr, dres_image, id, dres, ind, tracker)

if tracker.state ~= 1
    return;
else  % active
    
    % initialize the LK tracker
    tracker = LK_initialize(tracker, fr, id, dres, ind, dres_image);
    tracker.state = 2;
    tracker.streak_occluded = 0;
    tracker.streak_tracked = 0;
    
    % build the dres structure
    dres_one.fr = dres.fr(ind);
    dres_one.id = tracker.target_id;
    dres_one.x = dres.x(ind);
    dres_one.y = dres.y(ind);
    dres_one.w = dres.w(ind);
    dres_one.h = dres.h(ind);
    dres_one.r = dres.r(ind);
    dres_one.state = tracker.state;
    tracker.dres = dres_one;
end

% track a target
function tracker = track(fr, dres_image, dres, tracker, opt)

% tracked
if tracker.state == 2
    tracker.streak_occluded = 0;
    tracker.streak_tracked = tracker.streak_tracked + 1;
    tracker = MDP_value(tracker, fr, dres_image, dres, []);
    
    % check if target outside image
    [~, ov] = calc_overlap(tracker.dres, numel(tracker.dres.fr), dres_image, fr);
    if ov < opt.exit_threshold
        if opt.is_text
        end
        tracker.state = 0;
    end
end


% associate a lost target
function tracker = associate(fr, dres_image, dres_associate, tracker, opt)

% occluded
if tracker.state == 3
    tracker.streak_occluded = tracker.streak_occluded + 1;
    % find a set of detections for association
    [dres_associate, index_det] = generate_association_index(tracker, fr, dres_associate);
    tracker = MDP_value(tracker, fr, dres_image, dres_associate, index_det);
    if tracker.state == 2
        tracker.streak_occluded = 0;
    end
    
    if tracker.streak_occluded > opt.max_occlusion
        tracker.state = 0;
        if opt.is_text
        end
    end
    
    % check if target outside image
    [~, ov] = calc_overlap(tracker.dres, numel(tracker.dres.fr), dres_image, fr);
    if ov < opt.exit_threshold
        if opt.is_text
            
        end
        tracker.state = 0;
    end
end


% resolve conflict between trackers
function trackers = resolve(trackers, dres_det, opt)

% collect dres from trackers
dres_track = [];
for i = 1:numel(trackers)
    tracker = trackers{i};
    dres = sub(tracker.dres, numel(tracker.dres.fr));
    
    if tracker.state == 2
        if isempty(dres_track)
            dres_track = dres;
        else
            dres_track = concatenate_dres(dres_track, dres);
        end
    end
end

% compute overlaps
num_det = numel(dres_det.fr);
if isempty(dres_track)
    num_track = 0;
else
    num_track = numel(dres_track.fr);
end

flag = zeros(num_track, 1);
for i = 1:num_track
    [~, o] = calc_overlap(dres_track, i, dres_track, 1:num_track);
    o(i) = 0;
    o(flag == 1) = 0;
    [mo, ind] = max(o);
    if mo > opt.overlap_sup
        num1 = trackers{dres_track.id(i)}.streak_tracked;
        num2 = trackers{dres_track.id(ind)}.streak_tracked;
        o1 = max(calc_overlap(dres_track, i, dres_det, 1:num_det));
        o2 = max(calc_overlap(dres_track, ind, dres_det, 1:num_det));
        
        if num1 > num2
            sup = ind;
        elseif num1 < num2
            sup = i;
        else
            if o1 > o2
                sup = ind;
            else
                sup = i;
            end
        end
        
        trackers{dres_track.id(sup)}.state = 3;
        trackers{dres_track.id(sup)}.dres.state(end) = 3;
        if opt.is_text
            
        end
        flag(sup) = 1;
    end
end