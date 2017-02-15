require 'nn'
require 'dp'
require 'cutorch'
require 'hdf5'
require 'rnn'
require 'Flickr30k'
require 'SelectLastNonZero'
require 'LanguageModel'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a Recurrent Model')
cmd:text('Example:')
cmd:text('$> th rnn-visual-attention.lua > results.txt')
cmd:text('Options:')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

--[[ data ]]--
cmd:option('--dataset', 'Flickr30k', 'which dataset to use : Flickr30k | MSCOCO')
cmd:option('--input_h5','../utils/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('--input_json','../utils/data.json','path to the json file containing additional info and vocab')

cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
if opt.cuda then
	require 'optim'
	require 'cunn'
	cutorch.setDevice(opt.useDevice)
end
xp = torch.load(opt.xpPath)
if opt.cuda then
  xp:cuda()
else
  xp:float()
end

agent = xp:model():get(1):get(2):get(1):get(1):get(1)
-- agent = xp:model():get(1)
-- print(agent)
agent:insert(xp:model():get(1):get(1),1)
agent:remove(#agent-1)
agent:insert(nn.Transpose({1,2}):cuda(),#agent)

textualModule = agent:get(2):get(1)
textualModule:remove(#textualModule)
textualModule:remove(#textualModule)
textualModule:remove(#textualModule)
textualModule:add(nn.Replicate(100):cuda())
textualModule:add(nn.Squeeze(2):cuda())

visualModule = agent:get(2):get(2)
visualModule:remove(#visualModule)
visualModule:remove(#visualModule)
visualModule:remove(#visualModule)

print(agent)

if opt.dataset == 'Flickr30k' then
  local utils = require 'utils'
  local info = utils.read_json(opt.input_json)
  test_group = {}
  for i,img in pairs(info.images) do
    if img.split == 'test' then
        test_group[#test_group+1] = {id=i,filename=img.filename}
    end
  end

  local hdf5_file = hdf5.open(opt.input_h5, 'r')
  local label_start_ix = hdf5_file:read('label_start_ix'):all()
  local label_end_ix = hdf5_file:read('label_end_ix'):all()
  local label_length = hdf5_file:read('label_length'):all()
  require 'hdf5'
  local hdf5_bboxes = hdf5.open('../proposals/Flickr30kEntities/selective_search/Proposals.hdf5', 'r')

    for k,v in pairs(test_group) do
        local ix1 = label_start_ix[v.id]
        local ix2 = label_end_ix[v.id]
        local boxes = hdf5_bboxes:read(v.filename..'.jpg'):all()
        local vis = hdf5_file:read('VGG-DET'):partial({v.id,v.id},{1,100},{1,4096}):resize(100,4096)
        for idx=ix1,ix2 do
            local text = hdf5_file:read('labels'):partial({idx,idx},{1,10})
            local confidences = agent:forward({text,vis})
            confidences, sorted_idx = torch.sort(confidences)
            local phrase_name = info.phrase_names[idx]
            -- print(v.filename, phrase_name)
            local file = io.open('../predictions/fast/'..phrase_name..'.txt',"w")
            for i=1,100 do
            local line = confidences[{1,i}]..' '
                         ..boxes[sorted_idx[{1,i}]][2]..' '
                         ..boxes[sorted_idx[{1,i}]][1]..' '..
                         boxes[sorted_idx[{1,i}]][4]..' '..
                         boxes[sorted_idx[{1,i}]][3]..'\n'
            file:write(line)
            end
            file:close()
        end
        if k % 100 == 0 then
            print('processing '..k..'/'..#test_group)
        end
    end
    hdf5_file:close()
end