require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('lfs')
require('sys')

npy4th = require('npy4th')

mgr = {}
include('train_semi.lua')

printf = utils.printf
mgr.model_path = 'models/'
