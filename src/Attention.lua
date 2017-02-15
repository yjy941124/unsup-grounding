local Attention, parent = torch.class('nn.Attention', 'nn.Module')

function Attention:__init(size)
   parent.__init(self)

   self.weight = torch.Tensor(size)
   self.gradWeight = torch.Tensor(size)
   self.SoftMax = nn.SoftMax():cuda()
   self:reset()

   self.W_h = nn.LinearNoBias(opt.rnn_size, opt.att_size)
   self.W_v = nn.LinearNoBias(4096, opt.att_size)
   self.CAddTable = nn.CAddTable()
   self.b_1 = nn.Add(opt.att_size)
   self.L_2 = nn.Linear(opt.att_size, 1)
end


function Attention:reset(stdv)
   -- if stdv then
   --    stdv = stdv * math.sqrt(3)
   -- else
   --    stdv = 1./math.sqrt(self.weight:size(1))
   -- end

   -- self.weight:uniform(-stdv, stdv);
   self.weight:fill(1)
end

function Attention:updateOutput(input)
   local text = input[1]
   local img = input[2]

   local W_ho = self.W_h:updateOutput(text)
   local W_vo = self.W_v:updateOutput(img)

   
   local L_1o = self.CAddTable()


   self.output:resize(img:size(1),img:size(3)):zero()
   local alpha = self.SoftMax:updateOutput(self.weight)
   for i=1,img:size(1) do
      for j=1,img:size(2) do
         self.output[i]:add(alpha[i],img[{i,j}])
      end
   end
   return self.output
end

function Attention:updateGradInput(input, gradOutput)
   local text = input[1]
   local img = input[2]
   self.gradInput:resizeAs(input):zero()
   for i=1,img:size(1) do
      self.gradInput[i]:add(self.SoftMax.output[i], gradOutput)
   end
   return self.gradInput
end

function Attention:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local grad = torch.CudaTensor()
   grad:resizeAs(self.weight):zero()
   for i=1,input:size(1) do
      grad[i] = input[i]:dot(gradOutput)
   end
   self.gradWeight:add(self.SoftMax:updateGradInput(self.weight, grad))
end