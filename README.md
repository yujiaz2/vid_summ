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
   
    step 1:
    replace $TENSORFLOW/python/ops/rnn_cell.py to modify the activation function for KL divergence
    
    step 2:
    run train_1_layer.py to train the model
    
    step 3:
    run train_1_layer.py to get the max and min reconstruction error of offline training
    
    step 4:
    run update_1_layer.py using max and min offline training error to adaptively online update the model
    
    
## 10. sparsecoding.m
   * implement basic sparse coding algorithm
   
   
## 11. evaluation
   * get the evaluation results for algorithms
   
    run Eval_final.m
