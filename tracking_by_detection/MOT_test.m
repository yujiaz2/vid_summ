% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
%
% test on MOT benchmark
function MOT_test2

is_train = 0;
opt = globals();

mot2d_train_seqs = {'TUD-Stadtmitte', 'TUD-Campus', 'PETS09-S2L1', ...
    'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2', 'ADL-Rundle-6', ...
    'ADL-Rundle-8', 'KITTI-13', 'KITTI-17', 'Venice-2'};

mot2d_test_seqs = {'TUD-Crossing', 'PETS09-S2L2', 'ETH-Jelmoli', ...
    'ETH-Linthescher', 'ETH-Crossing', 'AVG-TownCentre', 'ADL-Rundle-1', ...
    'ADL-Rundle-3', 'KITTI-16', 'KITTI-19', 'Venice-1'};

% training and testing pairs
seq_idx_train = {{1, 2}, {3},    {4, 5, 6}, {7, 8}, {9, 10}, {11}};
seq_idx_test  = {{1},    {2, 6}, {3, 4, 5}, {7, 8}, {9, 10}, {11}};
seq_set_test = 'test';
N = numel(seq_idx_train);

test_time = 0;

if is_train
    
    num = numel(idx_train);
    tracker = [];
    for j = 1:num
        fprintf('Training on sequence: %s\n', mot2d_train_seqs{idx_train{j}});
        tracker = MDP_train(idx_train{j}, tracker);
        fprintf('%d training examples after training on %s\n', ...
            size(tracker.f_occluded, 1), mot2d_train_seqs{idx_train{j}});
    end
else
    % load tracker from file
    filename = sprintf('%s/TUD-Campus_tracker.mat', opt.results);
    object = load(filename);
    tracker = object.tracker;
    fprintf('load tracker from file %s\n', filename);
end

MDP_test(1, 1, tracker);

fprintf('Total time for testing: %f\n', test_time);