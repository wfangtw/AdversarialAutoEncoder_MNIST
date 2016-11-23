local train_mgr = torch.class('mgr.semi_mgr')

function train_mgr:__init(config)
  self.cuda = config.cuda
  self.y_dim = config.y_dim                       or 10
  self.z_dim = config.z_dim                       or 20
  self.in_dim = config.in_dim
  self.batch_size = config.batch_size             or 100
  self.epochs = config.epochs                     or 1000
  self.hidden = config.hidden                     or 1000
  self.dataset = config.dataset                   or 'mnist'
  self.iter_per_epoch = config.iter_per_epoch     or 500
  self.reg = 1e-4

  self.ae_lr = (self.dataset == 'cifar10')  and 0.001   or 0.005
  self.sup_lr = (self.dataset == 'cifar10') and 0.08     or 0.05
  self.adv_lr = (self.dataset == 'cifar10') and 0.01    or 0.05
  self.enc_reg_lr = self.adv_lr
  self.optimizer = 'sgd' -- 'adam'
  self.recon_enc_optim_params = {learningRate = self.ae_lr, momentum=0.9}
  self.recon_dec_optim_params = {learningRate = self.ae_lr, momentum=0.9}
  self.reg_enc_optim_params = {learningRate = self.enc_reg_lr, momentum=0.5}
  self.reg_top_optim_params = {learningRate = self.adv_lr, momentum=0.8}
  self.reg_bottom_optim_params = {learningRate = self.adv_lr, momentum=0.1}
  self.sup_optim_params = {learningRate = self.sup_lr, momentum=0.9}

  self.encoder = self:get_encoder()
  self.decoder = self:get_decoder()
  self.top_adv_net = self:get_adv_net(self.y_dim)
  self.bottom_adv_net = self:get_adv_net(self.z_dim)

  self.ae_crit = self.cuda and
    nn.MSECriterion():cuda() or
    nn.MSECriterion()

  self.adv_crit = self.cuda and
    nn.BCECriterion():cuda() or
    nn.BCECriterion()

  self.sup_crit = self.cuda and
    nn.CrossEntropyCriterion():cuda() or
    nn.CrossEntropyCriterion()

  --local ae = nn.Sequential()
    --:add(self.encoder)
    --:add(self.decoder)
  --self.params, self.grad_params = ae:getParameters()
  self.enc_params, self.enc_grad_params = self.encoder:getParameters()
  self.dec_params, self.dec_grad_params = self.decoder:getParameters()
  self.top_adv_params, self.top_adv_grad_params = self.top_adv_net:getParameters()
  self.bottom_adv_params, self.bottom_adv_grad_params = self.bottom_adv_net:getParameters()
end

function train_mgr:get_encoder()
  local top, bottom
  local input = nn.Identity()()
  if self.dataset == 'mnist' then
    --local inter = nn.Dropout(0.2)(input)
    --inter = nn.Linear(self.in_dim, self.hidden)(inter)
    local inter = nn.Linear(self.in_dim, self.hidden)(input)
    --inter = nn.BatchNormalization(self.hidden)(inter)
    inter = nn.ReLU()(inter)
    inter = nn.Linear(self.hidden, self.hidden)(inter)
    --inter = nn.BatchNormalization(self.hidden)(inter) inter = nn.ReLU()(inter)

    top = nn.Linear(self.hidden, self.y_dim)(inter)
    --top = nn.BatchNormalization(self.y_dim)(top)
    top = nn.SoftMax()(top)
    bottom = nn.Linear(self.hidden, self.z_dim)(inter)
    --bottom = nn.BatchNormalization(self.z_dim)(bottom)

    --inter = nn.Linear(self.hidden, self.z_dim + self.y_dim)(inter)
    --local top = nn.Narrow(2, 1, self.y_dim)(inter) -- dim=2, start_idx=1, end_idx=y_dim
    --local bottom = nn.Narrow(2, self.y_dim + 1, self.z_dim)(inter) -- dim=2, start_idx=y_dim+1, end_idx=y_dim+z_dim
    --top = nn.SoftMax()(top)
  else
    -- in: 3x32x32
    local inter = nn.Dropout(0.2)(input)
    inter = nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1)(inter) -- out: 16x32x32
    inter = nn.SpatialBatchNormalization(16)(inter)
    inter = nn.ReLU()(inter)
    inter = nn.Dropout(0.4)(inter)
    inter = nn.SpatialMaxPooling(2, 2, 2, 2)(inter) -- out: 16x16x16
    inter = nn.SpatialConvolution(16, 64, 3, 3, 1, 1, 1, 1)(inter) -- out: 128x16x16
    inter = nn.SpatialBatchNormalization(64)(inter)
    inter = nn.ReLU()(inter)
    inter = nn.Dropout(0.4)(inter)
    inter = nn.SpatialMaxPooling(2, 2, 2, 2)(inter) -- out: 128x8x8
    inter = nn.View(64 * 8 * 8)(inter)

    top = nn.Linear(64 * 8 * 8, self.y_dim)(inter)
    top = nn.BatchNormalization(self.y_dim)(top)
    top = nn.SoftMax()(top)
    bottom = nn.Linear(64 * 8 * 8, self.z_dim)(inter)
    bottom = nn.BatchNormalization(self.z_dim)(bottom)
  end

  local enc = nn.JoinTable(2){top, bottom}

  local net = nn.gModule({input}, {enc})
  if self.cuda then
    net:cuda()
  end
  return net
end

function train_mgr:get_decoder()
  local net 
  if self.dataset == 'mnist' then
    net = nn.Sequential()
      :add(nn.Linear(self.z_dim + self.y_dim, self.hidden))
      --:add(nn.BatchNormalization(self.hidden))
      :add(nn.ReLU())
      :add(nn.Linear(self.hidden, self.hidden))
      --:add(nn.BatchNormalization(self.hidden))
      :add(nn.ReLU())
      :add(nn.Linear(self.hidden, self.in_dim))
      --:add(nn.BatchNormalization(self.in_dim))
      --:add(nn.ReLU())
      --:add(nn.Sigmoid())
  else
    net = nn.Sequential()
      :add(nn.Linear(self.z_dim + self.y_dim, 64 * 8 * 8))
      :add(nn.BatchNormalization(64 * 8 * 8))
      :add(nn.ReLU())
      :add(nn.Dropout(0.4))
      :add(nn.View(torch.LongStorage{64, 8, 8})) -- out: 128x8x8
      :add(nn.SpatialUpSamplingNearest(2)) -- out: 128x16x16
      :add(nn.SpatialConvolution(64, 16, 3, 3, 1, 1, 1, 1)) -- out: 16x16x16
      :add(nn.SpatialBatchNormalization(16))
      :add(nn.ReLU())
      :add(nn.Dropout(0.4))
      :add(nn.SpatialUpSamplingNearest(2)) -- out: 16x32x32
      :add(nn.SpatialConvolution(16, 3, 3, 3, 1, 1, 1, 1)) -- out: 3x32x32
      :add(nn.SpatialBatchNormalization(3))
      :add(nn.Sigmoid())
  end

  if self.cuda then
    net:cuda()
  end
  return net
end

function train_mgr:get_adv_net(dim)
  local net = nn.Sequential()
    :add(nn.Linear(dim, self.hidden))
    :add(nn.ReLU())
    :add(nn.Linear(self.hidden, self.hidden))
    :add(nn.ReLU())
    :add(nn.Linear(self.hidden, 1))
    :add(nn.Sigmoid())
  if self.cuda then
    net:cuda()
  end
  return net
end

function train_mgr:train_reconstruction(x)
  local h = self.encoder:forward(x)
  local x_hat = self.decoder:forward(h)
  self.loss = self.ae_crit:forward(x_hat, x)
  local grad_loss = self.ae_crit:backward(x_hat, x)
  local grad_h = self.decoder:backward(h, grad_loss)
  self.encoder:backward(x, grad_h)
end

function train_mgr:train_regularization(x)
  local batch_size = x:size(1)
  local real_gauss = self.cuda and
    torch.Tensor(batch_size, self.z_dim):normal(0, 1):cuda() or
    torch.Tensor(batch_size, self.z_dim):normal(0, 1)
  local real_cat = self:get_real_cat(batch_size)
  local y_real = self.cuda and
    torch.ones(batch_size):cuda() or
    torch.ones(batch_size)
  local y_fake = self.cuda and
    torch.zeros(batch_size):cuda() or
    torch.zeros(batch_size)

  self.encoder:forward(x)
  self.loss = 0

  -- Train cat adversary to maximize log prob of real cat samples
  local pred = self.top_adv_net:forward(real_cat)
  local top_real_loss = self.adv_crit:forward(pred, y_real)
  local grad_real_loss = self.adv_crit:backward(pred, y_real):clone()
  self.top_adv_net:backward(real_cat, grad_real_loss)

  -- Train gauss adversary to maximize log prob of real gaussian samples
  pred = self.bottom_adv_net:forward(real_gauss)
  local bottom_real_loss = self.adv_crit:forward(pred, y_real)
  grad_real_loss = self.adv_crit:backward(pred, y_real):clone()
  self.bottom_adv_net:backward(real_gauss, grad_real_loss)

  -- Train cat adversary to minimize log prob of fake cat samples
  local top_output = self.encoder.output[{{},{1,self.y_dim}}]
  --print(top_output)
  local top_pred = self.top_adv_net:forward(top_output)
  local top_fake_loss = self.adv_crit:forward(top_pred, y_fake)
  local grad_fake_loss = self.adv_crit:backward(top_pred, y_fake):clone()
  self.top_adv_net:backward(top_output, grad_fake_loss)
  self.top_adv_loss = top_real_loss + top_fake_loss

  -- Train gauss adversary to minimize log prob of fake gaussian samples
  local bottom_output = self.encoder.output[{{},{-self.z_dim, -1}}]
  local bottom_pred = self.bottom_adv_net:forward(bottom_output)
  local bottom_fake_loss = self.adv_crit:forward(bottom_pred, y_fake)
  grad_fake_loss = self.adv_crit:backward(bottom_pred, y_fake):clone()
  self.bottom_adv_net:backward(bottom_output, grad_fake_loss)
  self.bottom_adv_loss = bottom_real_loss + bottom_fake_loss

  -- Train encoder to play minmax game with both adversary nets
  self.loss = self.loss + self.adv_crit:forward(top_pred, y_real)
  local top_grad_minmax_loss = self.adv_crit:backward(top_pred, y_real):clone()
  local top_grad_minmax = self.top_adv_net:updateGradInput(top_output, top_grad_minmax_loss):clone()--:mul(20)

  self.loss = self.loss + self.adv_crit:forward(bottom_pred, y_real)
  local bottom_grad_minmax_loss = self.adv_crit:backward(bottom_pred, y_real):clone()
  local bottom_grad_minmax = self.bottom_adv_net:updateGradInput(bottom_output, bottom_grad_minmax_loss):clone()
  
  local grad_minmax = torch.cat(top_grad_minmax, bottom_grad_minmax, 2)
  self.encoder:backward(x, grad_minmax)
end

function train_mgr:train_supervised(x, y)
  self.encoder:forward(x)

  local output = self.encoder.output[{{},{1,self.y_dim}}]:clone()
  local log = self.cuda and nn.Log():cuda() or nn.Log()
  local out = log:forward(output)
  self.loss = self.sup_crit:forward(out, y)
  local grad = self.sup_crit:backward(out, y)
  grad = log:backward(output, grad)
  local grad_zeros = self.cuda and 
    torch.zeros(self.batch_size, self.z_dim):cuda() or
    torch.zeros(self.batch_size, self.z_dim)
  grad = torch.cat(grad, grad_zeros, 2)
  self.encoder:backward(x, grad)
end

function train_mgr:sample_unlabeled(unlabeled_X)
  local sample_idx = torch.randperm(unlabeled_X:size(1))[{{1, self.batch_size}}]:long()
  local x = unlabeled_X:index(1, sample_idx)
  return x
end

function train_mgr:sample_labeled(train_X, train_y)
  local sample_idx = torch.randperm(train_X:size(1))[{{1, self.batch_size}}]:long()
  local x = train_X:index(1, sample_idx)
  local y = train_y:index(1, sample_idx)
  return x, y
end

function train_mgr:train(train_X, train_y, val_X, val_y, unlabeled_X)
  local x, y

  self.encoder:training()
  local _, loss
  local recon_losses, top_adv_losses, bottom_adv_losses = {}, {}, {}
  local reg_enc_losses, sup_losses = {}, {}
  --self.adv_lr = 0

  total_train_num = self.iter_per_epoch * self.batch_size
  local old_adv_lr, old_sup_lr
  for epoch = 1, self.epochs do
    printf('Epoch: %d\n', epoch)

    --if epoch == 1 then
      --old_adv_lr = self.adv_lr
      --old_sup_lr = self.sup_lr
      --self.adv_lr = 0
      --self.sup_lr = 0
    --elseif epoch == 2 then
      --self.adv_lr = old_adv_lr
      --self.sup_lr = old_sup_lr
    --end

    for n = 1, self.iter_per_epoch do
      xlua.progress((n - 1) * self.batch_size + 1, total_train_num)

      unlabeled_x = self:sample_unlabeled(unlabeled_X)
      labeled_x, labeled_y = self:sample_labeled(train_X, train_y)

      if self.dataset == 'mnist' then
        noise_unlabeled = self.cuda and
          torch.Tensor(unlabeled_x:size()):normal(0, 0.3):cuda() or
          torch.Tensor(unlabeled_x:size()):normal(0, 0.3)
        noise_labeled = self.cuda and
          torch.Tensor(labeled_x:size()):normal(0, 0.3):cuda() or
          torch.Tensor(labeled_x:size()):normal(0, 0.3)
        unlabeled_x = unlabeled_x + noise_unlabeled
        labeled_x = labeled_x + noise_labeled
      end

      local recon_enc_feval = function(params)
        --if self.params ~= params then
          --self.params:copy(params)
        --end
        self.enc_grad_params:zero()
        self.dec_grad_params:zero()

        self:train_reconstruction(unlabeled_x)
        self.enc_grad_params:add(self.reg, self.enc_params)
        return self.loss, self.enc_grad_params
      end

      local recon_dec_feval = function(params)
        self.dec_grad_params:add(self.reg, self.dec_params)
        return self.loss, self.dec_grad_params
      end

      local reg_feval = function(params)
        --if self.params ~= params then
          --self.params:copy(params)
        --end
        self.enc_grad_params:zero()
        self.top_adv_grad_params:zero()
        self.bottom_adv_grad_params:zero()

        self:train_regularization(unlabeled_x)
        self.enc_grad_params:add(self.reg, self.enc_params)
        return self.loss, self.enc_grad_params
      end

      local top_adv_feval = function(params)
        --if self.top_adv_params ~= params then
          --self.top_adv_params:copy(params)
        --end
        return self.top_adv_loss, self.top_adv_grad_params
      end

      local bottom_adv_feval = function(params)
        --if self.bottom_adv_params ~= params then
          --self.bottom_adv_params:copy(params)
        --end
        return self.bottom_adv_loss, self.bottom_adv_grad_params
      end

      local sup_feval = function(params)
        --if self.params ~= params then
          --self.params:copy(params)
        --end
        self.enc_grad_params:zero()
        self:train_supervised(labeled_x, labeled_y)
        self.enc_grad_params:add(self.reg, self.enc_params)
        return self.loss, self.enc_grad_params
      end

      _, loss = optim[self.optimizer](recon_enc_feval, self.enc_params, self.recon_enc_optim_params)
      recon_losses[#recon_losses + 1] = loss[1]

      optim[self.optimizer](recon_dec_feval, self.dec_params, self.recon_dec_optim_params)

      _, loss = optim[self.optimizer](reg_feval, self.enc_params, self.reg_enc_optim_params)
      reg_enc_losses[#reg_enc_losses + 1] = loss[1]

      _, loss = optim[self.optimizer](top_adv_feval, self.top_adv_params, self.reg_top_optim_params)
      top_adv_losses[#top_adv_losses + 1] = loss[1]

      _, loss = optim[self.optimizer](bottom_adv_feval, self.bottom_adv_params, self.reg_bottom_optim_params)
      bottom_adv_losses[#bottom_adv_losses + 1] = loss[1]

      _, loss = optim[self.optimizer](sup_feval, self.enc_params, self.sup_optim_params)
      sup_losses[#sup_losses + 1] = loss[1]

      --if n % 5 == 0 then
        --_, loss = optim[self.optimizer](sup_feval, self.enc_params, self.sup_optim_params)
        --sup_losses[#sup_losses + 1] = loss[1]
      --end
    end
    xlua.progress(total_train_num, total_train_num)

    printf('Reconstruction loss %.4f\n', torch.mean(torch.Tensor(recon_losses)))
    printf('Top adv regularization loss %.4f\n', torch.mean(torch.Tensor(top_adv_losses)))
    printf('Bottom adv regularization loss %.4f\n', torch.mean(torch.Tensor(bottom_adv_losses)))
    --printf('Encoder regularization loss %.4f\n', torch.mean(torch.Tensor(reg_enc_losses)))
    printf('Supervised loss %.4f\n', torch.mean(torch.Tensor(sup_losses)))

    local train_pred = self:predict(train_X)
    local acc = torch.eq(train_pred, train_y:int()):sum() / train_y:size(1)
    printf('Train acc: %.4f\n', acc)
    
    local val_pred = self:predict(val_X)
    acc = torch.eq(val_pred, val_y):sum() / val_y:size(1)
    printf('Val acc: %.4f\n', acc)

    recon_losses = {}
    top_adv_losses = {}
    bottom_adv_losses = {}
    reg_enc_losses = {}
    sup_losses = {}
    if epoch == 15 then
      self.enc_reg_lr = self.enc_reg_lr * 10
      self.adv_lr = self.adv_lr * 0.1
      --self.sup_lr = self.sup_lr * 5
    end
    --if epoch == 250 then
      --self.ae_lr = self.ae_lr * 0.1
      --self.adv_lr = self.adv_lr * 0.1
      --self.sup_lr = self.sup_lr * 0.1
    --end

    if epoch % 100 == 0 then
      local file_idx = 1
      while true do
        path = string.format(mgr.model_path .. '%s.z%d.hidden%d.iter%d.%d', self.dataset, self.z_dim, self.hidden, epoch, file_idx)
        if lfs.attributes(path) == nil then
          break
        end
        file_idx = file_idx + 1
      end
      print('Writing model to ' .. path)
      self:save(path)
    end
  end
end

function train_mgr:predict(X)
  self.encoder:evaluate()
  local dataset_size = X:size(1)
  local prediction = torch.IntTensor(dataset_size)
  for n = 1, dataset_size, self.batch_size do
    xlua.progress(n, dataset_size)
    local batch_size = (n + self.batch_size > dataset_size) and dataset_size - n or self.batch_size
    local x = X:narrow(1, n, batch_size)
    local output = self.encoder:forward(x)[{{}, {1, self.y_dim}}]
    local _, pred = torch.max(output, 2)
    pred = pred:int()
    prediction[{{n, n + batch_size - 1}}] = pred
  end
  xlua.progress(dataset_size, dataset_size)
  return prediction
end

function train_mgr:get_real_cat(batch_size)
  local cat = self.cuda and
    torch.zeros(batch_size, self.y_dim):cuda() or
    torch.zeros(batch_size, self.y_dim)
  for i = 1, batch_size do
    local num = torch.random(1, self.y_dim)
    cat[i][num] = 1
  end
  return cat
end

function train_mgr:save(path)
  local config = {
    cuda = self.cuda,
    y_dim = self.y_dim,
    z_dim = self.z_dim,
    in_dim = self.in_dim,
    batch_size = self.batch_size,
    epochs = self.epochs,
    hidden = self.hidden,
    dataset = self.dataset
  }
  torch.save(path, {
    enc_params = self.enc_params,
    dec_params = self.dec_params,
    top_adv_params = self.top_adv_params,
    bottom_adv_params = self.bottom_adv_params,
    config = config,
  })
end

function train_mgr.load(path)
  local state = torch.load(path)
  local model = mgr.train_mgr.new(state.config)
  model.enc_params:copy(state.enc_params)
  model.dec_params:copy(state.dec_params)
  model.top_adv_params:copy(state.top_adv_params)
  model.bottom_adv_params:copy(state.bottom_adv_params)
  return model
end
