------------------------------------------------------------------
--[[ SequenceSet ]]--
-- A DataSet for sequence classification in a hdf5 structure :
-- [data_path]/[class]/[imagename].JPEG  (folder-name is class-name)
-- Optimized for extremely large datasets (14 million images+).
-- Tested only on Linux (as it uses command-line linux utilities to 
-- scale up to 14 million+ images)
-- Images on disk can have different height, width and number of channels.
------------------------------------------------------------------------

local SequenceSet, parent = torch.class("dp.SequenceSet","dp.DataSet")

SequenceSet._input_shape = 'bt'
SequenceSet._output_shape = 'bt'

maxLength = 5

function SequenceSet:__init(config)
	assert(type(config) == 'table', "Constructor requires key-value arguments")
	self._args, self._groups, self._hdf5_file, which_set, 
	self._verbose = xlua.unpack(
		{config},
		'SequenceSet',
		'A DataSet for features in a structured dat file',
		{arg='groups', type='table', req=true,
		 help='one or many paths of sequences'},
		{arg='hdf5_file', type='table', req=true,
		 help='dataset file'},
		{arg='which_set',type='string', req=true,
		 default='train',
		  help='"train", "valid" or "test" set'},
		{arg='verbose', type='boolean', default=true,
		 help='display verbose messages'}
		)

	-- locals
	self:whichSet(which_set)

	self.label_start_ix = self._hdf5_file:read('label_start_ix'):all()
	self.label_end_ix = self._hdf5_file:read('label_end_ix'):all()
	self.label_length = self._hdf5_file:read('label_length'):all()

	local sum = 0
	for i=1,#self._groups do
		sum = sum + self.label_end_ix[self._groups[i]] - self.label_start_ix[self._groups[i]]+1
	end
	print('number of phrases in '..which_set..':', sum)

	-- buffers
	self._seqBuffers = {}

	-- required for multi-threading
	self._config = config
end

function SequenceSet:nSample()
	return #self._groups
end

function SequenceSet:sub(batch, start, stop)
	if not stop then
		stop = start
		start = batch
		batch = nil
	end

	batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}

	local textualTable = {}
	local visualTable = {}
	for idx = start, stop do
		-- load the sample
		local text = self._seqBuffers[(idx-start)*2+1] or torch.IntTensor()
		local vis = self._seqBuffers[(idx-start+1)*2] or torch.FloatTensor()
		text, vis = self:loadSequence(self._groups[idx])
		table.insert(textualTable, text)
		table.insert(visualTable, vis)		
	end

	local inputView = batch and batch:inputs() or dp.ListView{dp.ClassView(),dp.SequenceView()}
	local targetView = batch and batch:targets() or dp.ClassView()
	local textualTensor = inputView:components()[1]:input() or torch.IntTensor()
	local visualTensor = inputView:components()[2]:input() or torch.FloatTensor()

	self:tableToTensor(textualTable, visualTable, textualTensor, visualTensor)

	inputView:components()[1]:forward('bt', textualTensor)
	inputView:components()[2]:forward('bf', visualTensor)
	targetView:forward('bt', textualTensor)
	batch:inputs(inputView)
	batch:targets(targetView)
	return batch
end


function SequenceSet:index(batch, indices)
	if not indices then
	  indices = batch
	  batch = nil
	end
	batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}

	local textualTable = {}
	local visualTable = {}
	for i = 1, indices:size(1) do
		local idx = indices[i]
		-- load the sample
		-- load the sample
		local text = self._seqBuffers[i*2-1] or torch.IntTensor()
		local vis = self._seqBuffers[i*2] or torch.FloatTensor()
		text, vis = self:loadSequence(self._groups[idx])
		table.insert(textualTable, text)
		table.insert(visualTable, vis)		
	end

	local inputView = batch and batch:inputs() or dp.ListView{dp.ClassView(),dp.SequenceView()}
	local targetView = batch and batch:targets() or dp.ClassView()
	local textualTensor = inputView:components()[1]:input() or torch.IntTensor()
	local visualTensor = inputView:components()[2]:input() or torch.FloatTensor()

	self:tableToTensor(textualTable, visualTable, textualTensor, visualTensor)

	inputView:components()[1]:forward('bt', textualTensor)
	inputView:components()[2]:forward('bf', visualTensor)
	targetView:forward('bt', textualTensor)
	batch:inputs(inputView)
	batch:targets(targetView)

   return batch
end

-- converts a table of samples (and corresponding labels) to tensors
function SequenceSet:tableToTensor(textualTable, visualTable, textualTensor, visualTensor)
	textualTensor = textualTensor or torch.IntTensor()
	visualTensor = visualTensor or torch.FloatTensor()
	local n = #textualTable

	textualTensor:zeros(n*opt.seq_per_img, maxLength)
	visualTensor:zeros(n*opt.box_per_img, 4096)

	for i=1,n do
		textualTensor[{{(i-1)*opt.seq_per_img+1,i*opt.seq_per_img}}]:copy(textualTable[i])
		visualTensor[{{(i-1)*opt.box_per_img+1,i*opt.box_per_img}}]:copy(visualTable[i])
	end
	return textualTensor, visualTensor
end

function SequenceSet:loadSequence(index)
	local seq_per_img = opt.seq_per_img 
    local ix1 = self.label_start_ix[index]
    local ix2 = self.label_end_ix[index]
    local ncap = ix2 - ix1 + 1 -- number of captions available for this image
    assert(ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t')
    local text
    if ncap < seq_per_img then
    	-- we need to subsample (with replacement)
    	text = torch.LongTensor(seq_per_img, maxLength)
    	for q=1, seq_per_img do
    		local ixl = torch.random(ix1,ix2)
			text[{ {q,q} }] = self._hdf5_file:read('labels'):partial({ixl,ixl},{1,maxLength})
    	end
    else
    	-- there is enough data to read a contiguous chunk, but subsample the chunk position
    	local ixl = torch.random(ix1, ix2 - seq_per_img + 1) -- generates integer in the range
    	text = self._hdf5_file:read('labels'):partial({ixl, ixl+seq_per_img-1}, {1,maxLength})
    end
    local vis = self._hdf5_file:read('VGG-DET'):partial({index,index},{1,opt.box_per_img},{1,4096})

	return text, vis
end

function SequenceSet:getSequenceBuffer(i)
   self._seqBuffers[i] = self._seqBuffers[i] or torch.FloatTensor()
   return self._seqBuffers[i]
end

function SequenceSet:classes()
   return self._classes
end

------------------------ multithreading --------------------------------
function SequenceSet:multithread(nThread)
	nThread = nThread or 2

	local mainSeed = os.time()
	local config = self._config
	config.cache_mode = 'readonly'
	config.verbose = self._verbose

	local threads = require "threads"
	threads.Threads.serialization('threads.sharedserialize')
	self._threads = threads.Threads(
	  nThread,
	  function()
	     require 'dp'
	     require 'SequenceSet'
	  end,
	  function(idx)
	     opt = options -- pass to all donkeys via upvalue
	     tid = idx
	     local seed = mainSeed + idx
	     math.randomseed(seed)
	     torch.manualSeed(seed)
	     if config.verbose then
	        print(string.format('Starting worker thread with id: %d seed: %d', tid, seed))
	     end
	     dataset = dp.SequenceSet(config)
	  end
	)

	self._send_batches = dp.Queue() -- batches sent from main to threads
	self._recv_batches = dp.Queue() -- batches received in main from threads
	self._buffer_batches = dp.Queue() -- buffered batches

	-- public variables
	self.nThread = nThread
	self.isAsync = true
end

function SequenceSet:synchronize()
   self._threads:synchronize()
   while not self._recv_batches:empty() do
     self._buffer_batches:put(self._recv_batches:get())
   end
end

function SequenceSet:subAsyncPut(batch, start, stop, callback)
	if not batch then
	  batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(stop-start+1)
	end
	local input = batch:inputs():input()
	local target = batch:targets():input()
	assert(batch:inputs():input() and batch:targets():input())

	self._send_batches:put(batch)

	self._threads:addjob(
	  -- the job callback (runs in data-worker thread)
	function()
		tbatch = dataset:sub(start, stop)
	    input = tbatch:inputs():input()
	    target = tbatch:targets():input()
		return input, target
	end,
	  -- the endcallback (runs in the main thread)
	function(input, target)
		local batch = self._send_batches:get()
        batch:inputs():components()[1]:forward('bt', target[1])
        batch:inputs():components()[2]:forward('bf', target[2])
		batch:targets():forward('bt', input)
	     
		callback(batch)

		self._recv_batches:put(batch)
	end
	)
end

function SequenceSet:sampleAsyncPut(batch, nSample, sampleFunc, callback)
   self._iter_mode = self._iter_mode or 'sample'
   if (self._iter_mode ~= 'sample') then
      error'can only use one Sampler per async SequenceSet (for now)'
   end  
   
   if not batch then
      batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(nSample)
   end
   local input = batch:inputs():input()
   local target = batch:targets():input()
   assert(input and target)
   
   -- transfer the storage pointer over to a thread
   local inputPointer = tonumber(ffi.cast('intptr_t', torch.pointer(input:storage())))
   local targetPointer = tonumber(ffi.cast('intptr_t', torch.pointer(target:storage())))
   input:cdata().storage = nil
   target:cdata().storage = nil
   
   self._send_batches:put(batch)
   
   assert(self._threads:acceptsjob())
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         -- set the transfered storage
         torch.setFloatStorage(input, inputPointer)
         torch.setFloatStorage(target, targetPointer)
		 tbatch:inputs():components()[1]:forward('bt', input[1])
		 tbatch:inputs():components()[2]:forward('bf', input[2])
		 tbatch:targets():forward('bt', target)
         
         dataset:sample(tbatch, nSample, sampleFunc)
         
         -- transfer it back to the main thread
         local istg = tonumber(ffi.cast('intptr_t', torch.pointer(input:storage())))
         local tstg = tonumber(ffi.cast('intptr_t', torch.pointer(target:storage())))
         input:cdata().storage = nil
         target:cdata().storage = nil
         return input, target, istg, tstg
      end,
      -- the endcallback (runs in the main thread)
      function(input, target, istg, tstg)
         local batch = self._send_batches:get()
         torch.setFloatStorage(input, istg)
         torch.setFloatStorage(target, tstg)
		 batch:inputs():components()[1]:forward('bt', target[1])
		 batch:inputs():components()[2]:forward('bf', target[2])
		 batch:targets():forward('bt', input)
         
         callback(batch)
         
         self._recv_batches:put(batch)
      end
   )
end

-- recv results from worker : get results from queue
function SequenceSet:asyncGet()
	-- necessary because Threads:addjob sometimes calls dojob...
	if self._recv_batches:empty() then
		self._threads:dojob()
	end

	return self._recv_batches:get()
end