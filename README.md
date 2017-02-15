

# Unsupervised Grounding


#### 用 Selective Search 提取图片的 Proposals

基于http://koen.me/research/selectivesearch/ 修改的
```Shell
cd $ROOT/utils/SelectiveSearchCodeIJCV
matlab -nodisplay -nosplash
>> demoFlickr30kEntities
```
demoFlickr30kEntities.m中imgFolderPath指的是Flickr30kEntities图片存放的路径，需要自行修改。另外，因为.mat文件的存储空间有限，我每3000张存成一个.mat文件，然后用mergeBlob.m将多个文件整合成一个。生成的文件存储在$ROOT/proposals/Flickr30kEntities/selective_search/SelectiveSearch_Flickr30k_fast.mat

#### 用 Fast-RCNN 提取 Proposal 特征

基于https://github.com/facebookresearch/multipathnet 修改的
```Shell
cd $ROOT/utils/extract-VGG-DET
CUDA_VISIBLE_DEVICES=1 th extract-feature.lua
```
extract-feature.lua中的"../../proposals/Flickr30kEntities/selective_search/SelectiveSearch_Flickr30k_fast.mat"是上一步生成proposals的路径

#### 生成数据文件
这一步得到一个hdf5和一个json文件，用于下一步的训练。程序用到的文本信息存储在$ROOT/utils/GTPhrases_Flickr30kEntities.mat中。
```Shell
cd $ROOT/utils/
python generate_flickr30kEntities_dataset.py
```


#### 训练
input_h5和input_json分别对应上一步的两个文件，batch_size指一次迭代用几个图片训练，seq_per_img指每张图片对应一个phrase，minLR是最小的learningRate，maxWait是指如果有10个epoch验证集loss不降就减小learning Rate，maxTries指若有50个epoch验证集loss不降，就停止训练。

```Shell
cd $ROOT/src
CUDA_VISIBLE_DEVICES=0 th train.lua --input_h5 ../data/Flickr30kEntities/data.h5 --input_json ../data/Flickr30kEntities/data.json --batch_size 1 --seq_per_img 40 --learningRate 0.01 --minLR 0.00001 --maxWait 10 --maxTries 50 --cuda --progress
```
#### 测试
首先生成预测的结果。xpPath对应上一步生成的模型。
```Shell
cd $ROOT/src
CUDA_VISIBLE_DEVICES=3 th generate_predictions.lua --input_h5 ../data/Flickr30kEntities/data.h5 --input_json ../data/Flickr30kEntities/data.json --cuda --xpPath ../save/PASCAL:1485607794:1.dat
```
然后用http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/ 做evaluation。
注意要把PhraseLocalizationEval.m里的scoreOrdering改成'descend'。
