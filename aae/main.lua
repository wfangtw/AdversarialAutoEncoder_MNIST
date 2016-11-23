require('.')


local args = lapp [[
Training script for adversarial autoencoder (AAE)
    -m,--model  (default semi)          AAE model: {semi}
    -d,--data   (default mnist)         Dataset: {mnist}
    -y,--ydim   (default 10)						Number of classes
		-z,--zdim   (default 20)						Latent style dimension
    -e,--epochs	(default 10)            Number of training epochs
		-b,--batch	(default 100)						Batch size
    -o,--output (default blah)          Output numpy file path
    -c,--cuda                           Cuda
]]

local data_dir
if args.data == 'mnist' then
  data_dir = '../npy_mnist/'
else
  data_dir = '../npy_cifar10/'
end

local train_X = npy4th.loadnpy(data_dir .. 'train_X.npy') / 255
local val_X = npy4th.loadnpy(data_dir .. 'val_X.npy') / 255
local train_y = npy4th.loadnpy(data_dir .. 'train_y.npy'):int()
local val_y = npy4th.loadnpy(data_dir .. 'val_y.npy'):int()
local unlabeled_X = npy4th.loadnpy(data_dir .. 'unlabeled_X.npy') / 255.0

if args.data == 'mnist' then
  train_y = train_y + 1
  val_y = val_y + 1
else
  train_X = torch.reshape(train_X, train_X:size(1), 3, 32, 32)
  val_X = torch.reshape(val_X, val_X:size(1), 3, 32, 32)
  unlabeled_X = torch.reshape(unlabeled_X, unlabeled_X:size(1), 3, 32, 32)
end

if args.cuda then
	require('cutorch')
	require('cunn')
	train_X = train_X:cuda()
  train_y = train_y:cuda()
	val_X = val_X:cuda()
  unlabeled_X = unlabeled_X:cuda()
end

local model
if args.model == 'normal' then
  model = mgr.aae_mgr
elseif args.model == 'unsu' then
  model = mgr.train_mgr
else
  model = mgr.semi_mgr
end

local input_dim = train_X:size(2)

local autoencoder = model{
	y_dim = args.ydim,
	z_dim = args.zdim,
	cuda = args.cuda,
	epochs = args.epochs,
	batch_size = args.batch,
  dataset = args.data, 
	in_dim = input_dim,
}

if args.model == 'semi' then
  autoencoder:train(train_X, train_y, val_X, val_y, unlabeled_X)
  --local output = autoencoder:predict(val_X)
else
  autoencoder:train(X)
  local output = autoencoder:transform(X)
  npy4th.savenpy(args.output, output)
end

--local prediction = autoencoder:predict(X)

--local file = torch.DiskFile('../title_result/aae_res', 'w')
--for i = 1, prediction:size(1) do
	--file:writeInt(prediction[i])
--end
--file:close()
