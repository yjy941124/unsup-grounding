require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
require 'ImageDetect'
require 'image'
require 'ImageTransformer'
require 'hdf5'

local hdf5_detections = hdf5.open('VGG-DET.hdf5','w')
local hdf5_bboxes = hdf5.open('Proposals.hdf5','w')

local utils = require 'utils'
model = torch.load('../../models/vgg16_fast_rcnn_iter_40000.t7'):unpack():cuda()
model:remove(#model)
local transformer = ImageTransformer({102.9801,115.9465,122.7717}, nil, 255, {3,2,1})
local detec = ImageDetect(model,transformer,scale,max_size)

local matio = require 'matio'
matio.use_lua_strings = true
boxes = matio.load('../../proposals/Flickr30kEntities/selective_search/SelectiveSearch_Flickr30k_fast.mat','boxes')
imgNames = matio.load('../../proposals/Flickr30kEntities/selective_search/SelectiveSearch_Flickr30k_fast.mat','imgNames')

output = {}
for i=1,#boxes do
	local box = boxes[i]:float()[{{1,100},{1,4}}]
	local im = image.load('/home/zsy/data/flickr30k/flickr30k-images/'..imgNames[i])
	local output = detec:detect(im,box,1,true):float()
	hdf5_detections:write(imgNames[i],output)
	print(imgNames[i])
	hdf5_bboxes:write(imgNames[i],box)

	xlua.progress(i,#boxes)
end

hdf5_detections:close()
hdf5_bboxes:close()
