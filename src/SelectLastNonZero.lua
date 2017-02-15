local SelectLastNonZero, parent = torch.class('nn.SelectLastNonZero', 'nn.Module')

function SelectLastNonZero:__init()
	parent.__init(self)
	self.index = torch.IntTensor()
end

function SelectLastNonZero:updateOutput(input)
	-- input: batch_size x seq_len x dim
	self.output:resize(input:size(1),input:size(3))
	self.index:resize(input:size(1))
	for i=1,input:size(1) do
		for j=1,input:size(2) do
			if torch.sum(input[{i,j}]) == 0 then
				self.index[i] = j-1
				break
			end
		end
		self.output[i]:copy(input[{i,self.index[i]}]);
	end
	return self.output
end

function SelectLastNonZero:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(input)  
	self.gradInput:zero()
	for i=1,input:size(1) do
		self.gradInput[{i,self.index[i]}]:copy(gradOutput[i]) 
	end
	return self.gradInput
end