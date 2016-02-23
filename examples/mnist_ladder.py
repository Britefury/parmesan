"""
SEMI SUPERVISED LADDER NETWORKS ON MNIST
-----------------------------------------
This code reproduces the fully connected MNIST results on
with ladder networks.

See http://arxiv.org/abs/1507.02672


The code will get to 1.25 which is slightly worse than the results reported
in http://arxiv.org/abs/1507.02672 (1.25)


Note
-----
Because the batchnormalization collects statistics by running the
training data through the network in a single batch you'll probably need a
GPU with a large memory. The code is tested on TitanX and K40.

If you encounter memory problems you can try changing alpha from 'single_pass'
to eg. 0.5. You'll then need to run the training data through the collect
function in batches. This is completely untested (!).

Credits
-------
The batchnormalization code is based on code written by
Jan Schluter https://gist.github.com/f0k/f1a6bd3c8585c400c190



"""
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import identity, softmax
import numpy as np
import theano
import theano.tensor as T
from parmesan.layers import (ListIndexLayer, NormalizeLayer,
                             ScaleAndShiftLayer, DecoderNormalizeLayer,
                             DenoiseLayer,)
from parmesan.utils import theano_graph_hash_hex
from parmesan.layers.ladderlayers import RasmusInit, DenseLadderLayer, InputLadderLayer, build_ladder_ae
import os, sys, time
import uuid
import parmesan
import hashlib, binascii, cStringIO



filename_script = os.path.basename(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-lambdas", type=str,
                    default='1000,10,0.1,0.1,0.1,0.1,0.1')
parser.add_argument("-lr", type=str, default='0.001')
parser.add_argument("-optimizer", type=str, default='adam')
parser.add_argument("-init", type=str, default='None')
parser.add_argument("-initval", type=str, default='None')
parser.add_argument("-gradclip", type=str, default='1')
args = parser.parse_args()


num_classes = 10
batch_size = 100  # fails if batch_size != batch_size
collect_batch_size = 2000
num_labels = 100

np.random.seed(1234) # reproducibility

output_folder = os.path.join("results", os.path.splitext(filename_script)[0] + str(uuid.uuid4())[:18].replace('-', '_'))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file = os.path.join(output_folder, 'results.log')

with open(output_file, 'wb') as f:
    f.write("#"*80 + "\n")
    for name, val in sorted(vars(args).items()):
        s = str(name) + " "*(40-len(name)) + str(val)
        f.write(s + "\n")
    f.write("#"*80 + "\n")

optimizers = {'adam': lasagne.updates.adam,
              'adadelta': lasagne.updates.adadelta,
              'rmsprop': lasagne.updates.rmsprop,
              'sgd': lasagne.updates.sgd,
              'nag': lasagne.updates.nesterov_momentum
              }
optimizer = optimizers[args.optimizer]

if args.init == 'None':  # default to antti rasmus init
    init = RasmusInit()
else:
    if args.initval != 'None':
        # if `-initval` is not `'None'` use it as first argument to Lasange initializer
        initargs = [float(args.initval)]
    else:
        # use default arguments for Lasange initializers
        initargs = []

    inits = {'he': lasagne.init.HeUniform(*initargs),
             'glorot': lasagne.init.GlorotUniform(*initargs),
             'uniform': lasagne.init.Uniform(*initargs),
             'normal': lasagne.init.Normal(*initargs)}
    init = inits[args.init]


if args.gradclip == 'None':
    gradclip = None
else:
    gradclip = float(args.gradclip)

unit = lasagne.nonlinearities.leaky_rectify
lasagne.random.set_rng(np.random.RandomState(seed=1))

[x_train, targets_train, x_valid,
 targets_valid, x_test, targets_test] = parmesan.datasets.load_mnist_realval()

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
targets_train = targets_train.astype('int32')
targets_valid = targets_valid.astype('int32')
targets_test = targets_test.astype('int32')

np.random.seed(1)
shuffle = np.random.permutation(x_train.shape[0])
x_train_lab = x_train[:num_labels]
targets_train_lab = targets_train[:num_labels]
labeled_slice = slice(0, num_labels)
unlabeled_slice = slice(num_labels, 2*num_labels)

lambdas = map(float, args.lambdas.split(','))

assert len(lambdas) == 7
print "Lambdas: ", lambdas


num_classes = 10
num_inputs = 784
lr = float(args.lr)
noise = 0.3
num_epochs = 300
start_decay = 50
sym_x = T.matrix('sym_x')
sym_t = T.ivector('sym_t')
sh_lr = theano.shared(lasagne.utils.floatX(lr))



ll0 = InputLadderLayer(shape=(None, 28*28), cost_weight=lambdas[0])
ll1 = DenseLadderLayer(num_units_in=28*28, num_units_out=1000, nonlinearity=unit, init=init, cost_weight=lambdas[1])
ll2 = DenseLadderLayer(num_units_in=1000, num_units_out=500, nonlinearity=unit, init=init, cost_weight=lambdas[2])
ll3 = DenseLadderLayer(num_units_in=500, num_units_out=250, nonlinearity=unit, init=init, cost_weight=lambdas[3])
ll4 = DenseLadderLayer(num_units_in=250, num_units_out=250, nonlinearity=unit, init=init, cost_weight=lambdas[4])
ll5 = DenseLadderLayer(num_units_in=250, num_units_out=250, nonlinearity=unit, init=init, cost_weight=lambdas[5])
ll6 = DenseLadderLayer(num_units_in=250, num_units_out=10, nonlinearity=softmax, init=init, cost_weight=lambdas[6])


layers = [ll0, ll1, ll2, ll3, ll4, ll5, ll6]

costs, out_enc_clean, out_enc_noisy, collect_out = build_ladder_ae(layers, num_labels, unlabeled_slice, sym_x, sym_t,
                                                                   norm_alpha=0.1)

cost = sum(costs)

cost_hash_hex = theano_graph_hash_hex(cost)
print('Cost hash: {0}'.format(cost_hash_hex))
if cost_hash_hex != '0ef905eeec8ae028917bd06222c9e74295058dda85be8a630b7e6108635326ed':
    print('Cost function incorrect')

enc_out_clean_hash_hex = theano_graph_hash_hex(out_enc_clean)
print('enc_out_clean hash: {0}'.format(enc_out_clean_hash_hex))
if enc_out_clean_hash_hex != '498783576600b7f64b87cb79405232b3c591bb36ba4148d9b696318de50f8f58':
    print('enc_out_clean function incorrect')

out_enc_noisy_hash_hex = theano_graph_hash_hex(out_enc_noisy)
print('out_enc_noisy hash: {0}'.format(out_enc_noisy_hash_hex))
if out_enc_noisy_hash_hex != 'b3598519cf67f0d992aa078b2180c7917e51b2daca297aee03ad06c63582e239':
    print('out_enc_noisy function incorrect')

collect_out_hash_hex = theano_graph_hash_hex(collect_out)
print('collect_out hash: {0}'.format(collect_out_hash_hex))
if collect_out_hash_hex != 'db6d6c323b8e4f3b47f24426a791d7eb4903afe2680f8a40bdbf1b17be0dcc30':
    print('collect_out function incorrect')

# sys.exit()

# Get list of all trainable parameters in the network.
all_params = lasagne.layers.get_all_params(layers[0].l_z_hat_bn, trainable=True)
print ""*20 + "PARAMETERS" + "-"*20
for p in all_params:
    print p.name, p.get_value().shape
print "-"*60

if gradclip is not None:
    all_grads = [T.clip(g, -gradclip, gradclip)
                 for g in T.grad(cost, all_params)]
else:
    all_grads = T.grad(cost, all_params)

updates = optimizer(all_grads, all_params, learning_rate=sh_lr)

f_clean = theano.function([sym_x], out_enc_clean)

f_train = theano.function([sym_x, sym_t],
                          [cost, out_enc_noisy] + costs,
                          updates=updates, on_unused_input='warn')

# Our current implementation of batchnormalization collects the statistics
# by passing the entire training dataset through the network. This collects
# the correct statistics but is not possible for larger datasets...
f_collect = theano.function([sym_x],   # NO UPDATES !!!!!!! FOR COLLECT
                            [collect_out], on_unused_input='warn')

num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
loss = []


def train_epoch_semisupervised(x):
    confusion_train = parmesan.utils.ConfusionMatrix(num_classes)
    losses = []
    shuffle = np.random.permutation(x.shape[0])
    x = x[shuffle]
    for i in range(num_batches_train):
        idx = range(i*batch_size, (i+1)*batch_size)
        x_unsup = x[idx]

        # add labels
        x_batch = np.concatenate([x_train_lab, x_unsup], axis=0)

        # nb same targets all the time...
        output = f_train(x_batch, targets_train_lab)
        batch_loss, net_out = output[0], output[1]
        layer_costs = output[2:]
        # cut out preds with labels
        net_out = net_out[:num_labels]

        preds = np.argmax(net_out, axis=-1)
        confusion_train.batchadd(preds, targets_train_lab)
        losses += [batch_loss]
    collect()
    return confusion_train, losses, layer_costs


def collect():
    N = x_train.shape[0]
    n_collect_batches = N / collect_batch_size
    indices = np.arange(N)
    for i in range(n_collect_batches):
        batch_indices = indices[i*collect_batch_size:i*collect_batch_size+collect_batch_size]
        _ = f_collect(x_train[batch_indices])


def test_epoch(x, y):
    confusion_valid = parmesan.utils.ConfusionMatrix(num_classes)
    net_out = f_clean(x)
    preds = np.argmax(net_out, axis=-1)
    confusion_valid.batchadd(preds, y)
    return confusion_valid

with open(output_file, 'a') as f:
    f.write('Starting Training !\n')


for epoch in range(num_epochs):
    t1 = time.time()
    confusion_train, losses_train, layer_costs = train_epoch_semisupervised(x_train)
    confusion_valid = test_epoch(x_valid, targets_valid)
    confusion_test = test_epoch(x_test, targets_test)
    t2 = time.time()

    if any(np.isnan(losses_train)) or any(np.isinf(losses_train)):
        with open(output_file, 'w') as f:
            f.write('*NAN')
        break

    train_acc_cur = confusion_train.accuracy()
    valid_acc_cur = confusion_valid.accuracy()
    test_acc_cur = confusion_test.accuracy()

    if epoch > 3 and train_acc_cur < 0.1:
        with open(output_file, 'a') as f:
            f.write('*No progress')
        break

    if epoch > 30 and train_acc_cur < 0.5:
        with open(output_file, 'a') as f:
            f.write('*slow progres')
        break

    if epoch > start_decay:
        old_lr = sh_lr.get_value()
        new_lr = old_lr - (lr/(num_epochs-start_decay))
        sh_lr.set_value(lasagne.utils.floatX(new_lr))

    s = ("*EPOCH {}: time {:.2f}, tr-loss {:.6f}, tr-err {:.2f}%, va-err {:.2f}%, te-err {:.2f}%, lr {}").format(
        epoch, t2-t1, np.mean(losses_train), (1-train_acc_cur)*100.0, (1-valid_acc_cur)*100.0,
        (1-test_acc_cur)*100.0, sh_lr.get_value())
    print s
    with open(output_file, 'a') as f:
        f.write(s + "\n")

