require 'dp'
require 'SequenceSet'
require 'hdf5'

local Flikr30k, parent = torch.class("dp.Flikr30k", "dp.DataSource")
Flikr30k.isFlikr30k = true

Flikr30k._name = 'Flikr30k'

function Flikr30k:__init(config)
    config = config or {}
    assert(torch.type(config) == 'table', 
        "Constructor requires key-value arguments")

    self._args, self._hdf5_path, self._json_path = xlua.unpack(
        {config},
        'Flikr 30K',
        'http://shannon.cs.illinois.edu/DenotationGraph/',
        {arg='hdf5_path', type='string', 
         help='dataset file',
         default=''},
        {arg='json_path', type='string', 
         help='json file',
         default=''}
    )

    self:loadData()
    self:loadTrain()
    self:loadValid()
end

function Flikr30k:loadData()
    local utils = require 'utils'
    self.info = utils.read_json(self._json_path)
    self.vocab_size = utils.count_keys(self.info.ix_to_word)
    print('vocab size is ' .. self.vocab_size)

    self._train_group = {}
    self._valid_group = {}

    for i,img in pairs(self.info.images) do
        if img.split == 'train' then
            self._train_group[#self._train_group+1] = i
        elseif img.split == 'val' then
            self._valid_group[#self._valid_group+1] = i
        end
    end

    self.hdf5_file = hdf5.open(self._hdf5_path, 'r')
    local seq_size = self.hdf5_file:read('/labels'):dataspaceSize()
    self.seq_length = seq_size[2]
end

function Flikr30k:loadTrain()
    local dataset = dp.SequenceSet{
        groups = self._train_group,
        hdf5_file = self.hdf5_file,
        which_set = 'train'
    }
    self:trainSet(dataset)
    return dataset
end

function Flikr30k:loadValid()
    local dataset = dp.SequenceSet{
        groups = self._valid_group,
        hdf5_file = self.hdf5_file,
        which_set = 'valid'
    }
    self:validSet(dataset)
    return dataset
end

function Flikr30k:multithread(nThread)
   if self._train_set then
        self._train_set:multithread(nThread)
   end
   if self._valid_set then
        self._valid_set:multithread(nThread)
   end
end