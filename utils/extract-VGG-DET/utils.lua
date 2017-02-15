--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]
local utils = {}

function utils.convertFrom(out, bbox, y)
   if torch.type(out) == 'table' or out:nDimension() == 1 then
      local xc = (bbox[1] + bbox[3]) * 0.5
      local yc = (bbox[2] + bbox[4]) * 0.5
      local w = bbox[3] - bbox[1]
      local h = bbox[4] - bbox[2]

      local xtc = xc + y[1] * w
      local ytc = yc + y[2] * h
      local wt = w * math.exp(y[3])
      local ht = h * math.exp(y[4])

      out[1] = xtc - wt/2
      out[2] = ytc - ht/2
      out[3] = xtc + wt/2
      out[4] = ytc + ht/2
   else
      assert(bbox:size(2) == y:size(2))
      assert(bbox:size(2) == out:size(2))
      assert(bbox:size(1) == y:size(1))
      assert(bbox:size(1) == out:size(1))
      local xc = (bbox[{{},1}] + bbox[{{},3}]) * 0.5
      local yc = (bbox[{{},2}] + bbox[{{},4}]) * 0.5
      local w = bbox[{{},3}] - bbox[{{},1}]
      local h = bbox[{{},4}] - bbox[{{},2}]

      local xtc = torch.addcmul(xc, y[{{},1}], w)
      local ytc = torch.addcmul(yc, y[{{},2}], h)
      local wt = torch.exp(y[{{},3}]):cmul(w)
      local ht = torch.exp(y[{{},4}]):cmul(h)

      out[{{},1}] = xtc - wt * 0.5
      out[{{},2}] = ytc - ht * 0.5
      out[{{},3}] = xtc + wt * 0.5
      out[{{},4}] = ytc + ht * 0.5
   end
end

return utils