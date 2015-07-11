local ffi = require 'ffi'
local C = loadcaffe.C


loadcaffe.load = function(prototxt_name, binary_name, cuda_package)
  local cuda_package = cuda_package or 'nn'
  local handle = ffi.new('void*[1]')

  -- loads caffe model in memory and keeps handle to it in ffi
  local old_val = handle[1]
  C.loadBinary(handle, prototxt_name, binary_name)
  if old_val == handle[1] then return end

  -- transforms caffe prototxt to torch lua file model description and 
  -- writes to a script file
  local lua_name = prototxt_name..'.lua'
  C.convertProtoToLua(handle, lua_name, cuda_package)

  -- executes the script, defining global 'model' module list
  local model = dofile(lua_name)

  -- goes over the list, copying weights from caffe blobs to torch tensor
  local net = nn.Sequential()
  local list_modules = model
  for i,item in ipairs(list_modules) do
    item[2]:cuda()
    if item[2].weight then
      local w = torch.FloatTensor()
      local bias = torch.FloatTensor()
      C.loadModule(handle, item[1], w:cdata(), bias:cdata())
      if cuda_package == 'ccn2' then
        w = w:transpose(1,4):transpose(1,3):transpose(1,2)
      end

      if item[2].groups then
         -- Do the magic here
         local nb_groups = item[2].groups
         local input_group_size = w:size(2)
         local output_group_size = w:size(1) / nb_groups
         local grouped_w_size = w:size()

         grouped_w_size[2] = nb_groups * input_group_size
         local grouped_w = torch.FloatTensor(grouped_w_size)
         grouped_w:zero()

         for g=1, nb_groups do
            -- output_planes | input_planes | W | H
            local to_fill = grouped_w:narrow(1, 1+(g-1)*output_group_size, output_group_size)
                                     :narrow(2, 1+(g-1)*input_group_size, input_group_size)
            local filler = w:narrow(1, 1+(g-1)*output_group_size, output_group_size)
            to_fill:copy(filler)
         end

         w = grouped_w
      end


      item[2].weight:copy(w)
      item[2].bias:copy(bias)
    end
    net:add(item[2])
  end
  C.destroyBinary(handle)
  --print(net)
  return net
end
