# vid_summ
Diving into Moving Object Instances for Fine-Grained Video Summarization

## 1. vid_seq.py
   * get the video sequences for each video
    
    
## 2. detect_bbox_score/demo.py
   * get boundingboxes and scores for each video sequence
    
    step 1:
    download code from https://github.com/rbgirshick/py-faster-rcnn/
    
    step 2:
    replace https://github.com/rbgirshick/py-faster-rcnn/tree/master/tools/demo.py
    
    step 3:
    run py-faster-rcnn/tools/demo.py
  
    
## 3. tracking_by_detection
   * get the results of boundingboxes for each frame after online multi-object tracking by decision making
  
    step 1: 
    download code from https://github.com/yuxng/MDP_Tracking/
  
    step 2:
    replace https://github.com/yuxng/MDP_Tracking/MOT_test.m, https://github.com/yuxng/MDP_Tracking/MDP_test.m
    
    step 3:
    run MOT_test.m
    
    
## 4. obj_sequences.m
   * get the information of object sequences(first_fr_no, last_fr_no, bbox_img of the size of the boundingbox), and remove the still objects
    
    
## 5. superclip/demo.m, summe_superframeSegmentation.m
   * get the result of first_fr_no and last_fr_no on object superclips for each object sequence(object_level)
  
    step 1:
    download code from https://www.vision.ee.ethz.ch/~gyglim/vsum/
  
    step 2:
    replace https://www.vision.ee.ethz.ch/~gyglim/vsum/index.php#sf_code/demo.m, summe_superframeSegmentation.m
    
    step 3:
    run demo.m
    
    
## 6. obj_clip.m
   * get the result of object clip(all boundingboxes-x1,y1,w,h on each frame for object clips)
    
    
## 7. feature/demo.m, test.m
   * get features of each boundingboxes on object clip(obj_feature + location and context feature)
  
    step 1:
    download code from https://github.com/rbgirshick/fast-rcnn/
  
    step 2:
    replace https://github.com/rbgirshick/fast-rcnn/tools/demo.m, https://github.com/rbgirshick/fast-rcnn/lib/fast_rcnn/test.m
    
    step 3:
    run fast-rcnn/tools/demo.py
    
    
## 8. combine_all_feat.m
   * combine features of all object clips
    
    
## 9. ae
   * use online motion autoencoder to get the result of summarized key object motion clips (REFERENCE: https://github.com/iwyoo/LSTM-autoencoder/)
   
    We collect videos for training, validation and testing, and the features extracted are saved in *ae/find_iter/first_layer/feat_all_mean.mat*(offline train), *ae/find_iter/feat_validation_mean.mat*(validation) and *ae/online_feat*(testing).

    *Find_iter* is used for finding the parameter of iteration, *mAE* is used for online updating and obtaining the reconstruction loss, and *online_feat/feat_9parts.txt* stored the superclip segmentation results.

    step 1:
    replace $TENSORFLOW/python/ops/rnn_cell.py to modify the activation function for KL divergence
    
    step 2:
    run find_iter/train_1_layer.py to find the best value for parameter of iteration

    step 3:
    run mAE/train_1_layer.py to train the model.
    
    step 4:
    run mAE/train_1_layer_forward.py to get the max and min reconstruction error of offline training, and obtain the offline training date for next layer.
    
    step 5:
    run mAE/update_1_layer.py using max and min offline training error to adaptively online update the model

    step 6:
    run mAE/train_1_layer_validation_forward.py to get the evaluation data for next layer
    
    sterp 7:
    Repeat the above steps for other layers
    
    
## 10. sparsecoding.m
   * implement basic online sparse coding algorithm
   
   
## 11. evaluation
   * get the evaluation results for algorithms
   
    run Eval_final.m
