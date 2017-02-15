require 'dp'
require 'rnn'
require 'Flickr30k'
require 'Attention'
require 'LanguageModel'
require 'SelectLastNonZero'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an auto annotation model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('--input_h5','../utils/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('--input_json','../utils/data.json','path to the json file containing additional info and vocab')
cmd:option('--dataset', 'Flikr30k', 'which dataset to use : Flikr30k | MSCOCO')
cmd:option('--nThread', 0, 'allocate threads for loading features from disk. Requires threads-ffi.')

-- Model settings
cmd:option('--rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('--input_encoding_size',512,'the encoding size of each token in the vocabulary')
cmd:option('--input_visual_size',4096,'the encoding size of each token in the image.')
cmd:option('--att_size',512,'size of the rnn in number of hidden nodes in each layer')

-- Optimization: General
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('--lrDecay', 'adaptive', 'type of learning rate decay : adaptive | linear | schedule | none')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--maxWait', 20, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.5, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--minLR', 0.01, 'minimum learning rate')
cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batch_size', 1, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 2000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
cmd:option('--box_per_img',100,'number of bounding boxes per image')

-- misc
cmd:option('--backend', 'cudnn', 'nn|cudnn')
cmd:option('--seed', 123, 'random number generator seed to use')
cmd:option('--gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
opt = cmd:parse(arg)
if not opt.silent then
   table.print(opt)
end

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local ds = dp[opt.dataset]{hdf5_path = opt.input_h5, json_path = opt.input_json}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
  -- intialize language model
local lmOpt = {}
lmOpt.vocab_size = ds.vocab_size
lmOpt.input_encoding_size = opt.input_encoding_size
lmOpt.rnn_size = opt.rnn_size
lmOpt.num_layers = 1
lmOpt.dropout = opt.drop_prob_lm
lmOpt.seq_length = ds.seq_length
lmOpt.batch_size = opt.batch_size * opt.seq_per_img

agent = nn.Sequential()

-- create the core lstm network. note +1 for both the START and END tokens
texturalInput = - nn.Convert()
visualInput = - nn.Convert()

seqlstm = nn.SeqLSTM(opt.input_encoding_size, opt.rnn_size)
seqlstm.batchfirst = true
seqlstm.maskzero = true

texturalModule = texturalInput
                 - nn.LookupTableMaskZero(ds.vocab_size,opt.input_encoding_size)
                 - seqlstm
                 - nn.SelectLastNonZero()
                 - nn.LinearNoBias(opt.rnn_size, opt.att_size)
                 - nn.Reshape(opt.batch_size, opt.seq_per_img, opt.att_size)
                 - nn.Replicate(opt.box_per_img, 2, 2)
                 - nn.Reshape(opt.batch_size*opt.seq_per_img*opt.box_per_img, opt.att_size,false)

visualModule = visualInput
               - nn.LinearNoBias(opt.input_visual_size, opt.att_size)
               - nn.Reshape(opt.batch_size, opt.box_per_img, opt.att_size)
               - nn.Replicate(opt.seq_per_img, 2, 3)
               - nn.Reshape(opt.batch_size*opt.seq_per_img*opt.box_per_img, opt.att_size,false)

ext_visualInput = visualInput
                  - nn.Reshape(opt.batch_size, opt.box_per_img, opt.input_visual_size)
                  - nn.Replicate(opt.seq_per_img, 2, 3)
                  - nn.Reshape(opt.batch_size*opt.seq_per_img, opt.box_per_img, opt.input_visual_size,false)

attention = {texturalModule, visualModule}
            - nn.CAddTable()
            - nn.Add(opt.att_size)
            - nn.ReLU()
            - nn.Linear(opt.att_size,1)
            - nn.Reshape(opt.batch_size*opt.seq_per_img, 1, opt.box_per_img)
            - nn.SoftMax()

reconstruct = {attention, ext_visualInput}
              - nn.MM()
              - nn.Squeeze(2)
              - nn.Linear(opt.input_visual_size,opt.input_encoding_size)
              - nn.ReLU()

ext_texturalInput = texturalInput
                  - nn.Transpose({1,2})

LanguageModel = {reconstruct, ext_texturalInput}
                - nn.LanguageModel(lmOpt)

agent = nn.gModule({texturalInput, visualInput},{LanguageModel})

--[[Propagators]]--
if opt.lrDecay == 'adaptive' then
   ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
elseif opt.lrDecay == 'linear' then
   opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
end

targetModule = nn.Sequential()
targetModule:add(nn.Convert())
targetModule:add(nn.Transpose({1,2}))

train = dp.Optimizer{
   loss = nn.ModuleCriterion(nn.LanguageModelCriterion(),nil,targetModule) -- BACKPROP
   ,
   epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         if opt.lrDecay == 'adaptive' then
            opt.learningRate = opt.learningRate*ad.decay
            ad.decay = 1
         elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
            opt.learningRate = opt.schedule[report.epoch]
         elseif opt.lrDecay == 'linear' then 
            opt.learningRate = opt.learningRate + opt.decayFactor
         end
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
         end
      end
   end,
   callback = function(model, report)
      if opt.cutoffNorm > 0 then
         local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
         -- opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
         -- if opt.lastEpoch < report.epoch and not opt.silent then
         --    print("mean gradParam norm", opt.meanNorm)
         -- end         
      end
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams
   end,
   feedback = nil,
   sampler = dp.ShuffleSampler{
      epoch_size = -1, batch_size = opt.batch_size
   },
   progress = opt.progress
}


valid = dp.Evaluator{
   loss = nn.ModuleCriterion(nn.LanguageModelCriterion(),nil,targetModule),
   feedback = nil,
   sampler = dp.Sampler{epoch_size = -1, batch_size = opt.batch_size},
   progress = opt.progress
}

-- [[multithreading]]--
if opt.nThread > 0 then
   ds:multithread(opt.nThread)
   train:sampler():async()
   valid:sampler():async()
end

--[[Experiment]]--
xp = dp.Experiment{
   model = agent,
   optimizer = train,
   validator = valid,
   tester = nil,
   observer = {
      ad,
      dp.FileLogger('../log/'),
      dp.EarlyStopper{
         save_strategy = dp.SaveToFile{save_dir='../save/'},
         max_epochs = opt.maxTries,
         error_report={'validator','loss'}
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
else
   xp:float()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Agent :"
   print(agent)
end

xp.opt = opt
xp:run(ds)