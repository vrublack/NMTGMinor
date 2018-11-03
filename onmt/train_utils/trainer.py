from __future__ import division

import sys, tempfile
import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time, datetime
import random
import numpy as np
from onmt.multiprocessing.multiprocessing_wrapper import MultiprocessingRunner
from onmt.ModelConstructor import init_model_parameters
from onmt.train_utils.loss import HLoss
import translate


separator = '-' * 70


class BaseTrainer(object):

    def __init__(self, model, loss_function, trainData, validData, dataset, opt):

        self.model = model
        self.trainData = trainData
        self.validData = validData
        self.dicts = dataset['dicts']
        self.dataset = dataset
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1)

        self.loss_function = loss_function
        self.adv_loss_function = nn.NLLLoss(reduction='sum')

        self.start_time = 0

    def run(self, *args, **kwargs):

        raise NotImplementedError

    def eval(self, data):

        raise NotImplementedError

    def to_variable(self, data):

        for i, t in enumerate(data):
            if self.cuda:
                data[i] = Variable(data[i].cuda())
            else:
                data[i] = Variable(data[i])

        return data

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         'Use the param in the forward pass or set requires_grad=False')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset + numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, trainData, validData, dataset, opt):
        super().__init__(model, loss_function, trainData, validData, dataset, opt)
        self.optim = onmt.Optim(opt)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()

        self.optim.set_parameters(self.model.parameters())

    def save(self, epoch, valid_ppl, batchOrder=None, iteration=-1):

        opt, dataset = self.opt, self.dataset
        model = self.model

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'iteration': iteration,
            'batchOrder': batchOrder,
            'optim': optim_state_dict
        }

        file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        return file_name

    def eval(self, data):
        opt = self.opt
        w_reconstr, w_adv, w_classif = opt.w_reconstr, opt.w_adv, opt.w_classif

        epoch_loss = 0
        epoch_loss_reconstruction = 0
        epoch_loss_adv = 0
        total_words = 0
        nSamples = len(data)
        num_accumulated_sents = 0
        correct = 0

        batch_order = data.create_order()
        self.model.eval()
        """ New semantics of PyTorch: save space by not creating gradients """
        with torch.no_grad():
            for i in range(nSamples):

                samples = data.next()

                batch = self.to_variable(samples[0])

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """

                if int(batch[2][0].cpu().numpy()) == 0:
                    self.model.decoder.set_active(0)
                else:
                    self.model.decoder.set_active(1)

                outputs, classified_repr = self.model(batch)
                targets = batch[1][1:]
                targets_style = batch[2]

                batch_size = targets.size(1)

                loss_reconstruction, _ = self.loss_function(outputs, targets, generator=self.model.generator,
                                                            backward=False)
                loss_adv = self.adv_loss_function(classified_repr, targets_style)
                loss_total = w_reconstr * loss_reconstruction + w_classif * loss_adv

                loss_total = loss_total.data.cpu().numpy()
                loss_adv = loss_adv.data.cpu().numpy()
                loss_reconstruction = loss_reconstruction.data.cpu().numpy()

                epoch_loss += loss_total
                epoch_loss_reconstruction += loss_reconstruction
                epoch_loss_adv += loss_adv
                total_words += targets.data.ne(onmt.Constants.PAD).sum().item()
                num_accumulated_sents += batch_size
                correct += classified_repr.argmax(dim=1).eq(targets_style).sum(dim=0).cpu().numpy()


        self.model.train()
        return epoch_loss / total_words, epoch_loss_reconstruction / total_words, \
               epoch_loss_adv / num_accumulated_sents, correct / num_accumulated_sents

    def train_epoch(self, epoch, resume=False, batchOrder=None, iteration=0):

        opt = self.opt
        w_reconstr, w_adv, w_classif = opt.w_reconstr, opt.w_adv, opt.w_classif
        trainData = self.trainData

        if epoch <= opt.reconstr_headstart:
            # train reconstruction without adversarial loss
            self.model.set_trainable(True, True, False)
            train_phase = 'headstart'
        elif (epoch - opt.reconstr_headstart - 1) % (opt.reconstr_train_n + opt.classif_train_n) < opt.classif_train_n:
            # train discriminator
            self.model.set_trainable(False, False, True)
            train_phase = 'discriminator'
        else:
            # train reconstruction with adversarial loss
            self.model.set_trainable(True, True, False)
            train_phase = 'reconstruction'

        print('\nTraining phase: {}\n'.format(train_phase))

        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.

        if resume:
            trainData.batchOrder = batchOrder
            trainData.set_index(iteration)
            print("Resuming from iteration: %d" % iteration)
        else:
            batchOrder = trainData.create_order()
            iteration = 0
        # if batchOrder is not None:
        # batchOrder = trainData.create_order()
        # else:
        # trainData.batchOrder = batchOrder

        # if iteration is not None and iteration > -1:
        # trainData.set_index(iteration)
        # print("Resuming from iteration: %d" % iteration)

        epoch_loss = 0
        epoch_loss_reconstruction = 0
        epoch_loss_adv = 0
        total_words = 0
        report_loss, report_tgt_words = 0, 0
        correct = 0
        report_src_words = 0
        start = time.time()
        nSamples = len(trainData)
        dataset = self.dataset

        num_accumulated_sents = 0
        num_total_sents = 0

        for i in range(iteration, nSamples):

            curriculum = (epoch < opt.curriculum)

            samples = trainData.next(curriculum=curriculum)

            batch = self.to_variable(samples[0])

            oom = False

            if int(batch[2][0].cpu().numpy()) == 0:
                self.model.decoder.set_active(0)
            else:
                self.model.decoder.set_active(1)

            targets = batch[1][1:]
            targets_style = batch[2]

            batch_size = targets.size(1)

            try:
                def train_part(total_loss_f):
                    outputs, classified_repr = self.model(batch)

                    tgt_mask = targets.data.ne(onmt.Constants.PAD)
                    tgt_size = tgt_mask.sum()

                    tgt_mask = torch.autograd.Variable(tgt_mask)
                    # ~ tgt_mask = None
                    normalizer = 1

                    if self.opt.normalize_gradient:
                        normalizer = tgt_size

                    loss_reconstruction, _ = self.loss_function(outputs, targets, generator=self.model.generator,
                                                                           backward=False, mask=tgt_mask,
                                                                           normalizer=normalizer)

                    loss_adv = self.adv_loss_function(classified_repr, targets_style)
                    loss_total = total_loss_f(loss_reconstruction, loss_adv)
                    loss_total.backward()

                    # update parameters immediately
                    self.optim.step(grad_denom=1)
                    self.model.zero_grad()

                    return loss_total, loss_reconstruction, loss_adv, classified_repr

                if train_phase == 'headstart':
                    # train reconstruction without adversarial loss
                    total_loss_f = lambda loss_reconstr, loss_class: w_reconstr * loss_reconstr
                    loss_total, loss_reconstruction, loss_adv, classified_repr = train_part(total_loss_f)
                elif train_phase == 'reconstruction':
                    # train reconstruction with adversarial loss
                    total_loss_f = lambda loss_reconstr, loss_class: w_reconstr * loss_reconstr - w_adv * loss_class
                    loss_total, loss_reconstruction, loss_adv, classified_repr = train_part(total_loss_f)
                elif train_phase == 'discriminator':
                    # train discriminator
                    total_loss_f = lambda loss_reconstr, loss_class: w_classif * loss_class
                    loss_total, loss_reconstruction, loss_adv, classified_repr = train_part(total_loss_f)


            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                else:
                    raise e

            if not oom:
                num_accumulated_sents += batch_size
                num_total_sents += batch_size

                src_size = batch[0].data.ne(onmt.Constants.PAD).sum().item()
                tgt_size = targets.data.ne(onmt.Constants.PAD).sum().item()

                # important: convert to numpy (or set requires_grad to False), otherwise the statistics variables are tensors and contain
                # the history of the whole epoch, leading to a memory overflow
                loss_total = loss_total.data.cpu().numpy()
                loss_adv = loss_adv.data.cpu().numpy()
                loss_reconstruction = loss_reconstruction.data.cpu().numpy()

                num_words = tgt_size
                report_loss += loss_total
                report_tgt_words += num_words
                report_src_words += src_size
                epoch_loss += loss_total
                epoch_loss_reconstruction += loss_reconstruction
                epoch_loss_adv += loss_adv
                total_words += num_words
                correct += classified_repr.argmax(dim=1).eq(targets_style).sum(dim=0).cpu().numpy()

                optim = self.optim

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " +
                           "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
                          (epoch, i + 1, len(trainData),
                           math.exp(report_loss / report_tgt_words),
                           optim.getLearningRate(),
                           optim._step,
                           report_src_words / (time.time() - start),
                           report_tgt_words / (time.time() - start),
                           str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                    report_loss, report_tgt_words = 0, 0
                    report_src_words = 0
                    start = time.time()

        return epoch_loss / total_words, epoch_loss_reconstruction / total_words, \
               epoch_loss_adv / num_total_sents, correct / num_total_sents

    def run(self, save_file=None):

        opt = self.opt
        model = self.model
        dataset = self.dataset
        optim = self.optim

        # Try to load the save_file
        checkpoint = None
        if save_file:
            checkpoint = torch.load(save_file, map_location=lambda storage, loc: storage)

        if checkpoint is not None:
            print('Loading model and optim from checkpoint at %s' % save_file)
            self.model.load_state_dict(checkpoint['model'])

            if opt.reset_optim == False:
                self.optim.load_state_dict(checkpoint['optim'])
                batchOrder = checkpoint['batchOrder']
                iteration = checkpoint['iteration'] + 1
                opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))
                resume = True
            else:
                batchOrder = None
                iteration = 0
                resume = False

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            batchOrder = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False

        valid_loss, reconstr, adv, adv_accuracy = self.eval(self.validData)
        reconstr_ppl = math.exp(min(reconstr, 100))
        print(
            '{}\nValid loss: {}\nReconstruction ppl: {}\nAdv loss: {}, accuracy: {}\n{}\n'.format(separator,
                                                                                                                valid_loss,
                                                                                                                reconstr_ppl,
                                                                                                                adv, adv_accuracy,
                                                                                                                separator))
        self.start_time = time.time()

        best_train_loss = (None, None, None, None)
        best_val_loss = (None, None, None, None)

        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss, reconstr, adv, adv_accuracy = self.train_epoch(epoch, resume=resume,
                                                                batchOrder=batchOrder,
                                                                iteration=iteration)
            reconstr_ppl = math.exp(min(reconstr, 100))
            print('{}\nTrain loss: {}\nReconstruction ppl: {}\nAdv loss: {}, accuracy: {}\n{}\n'.format(separator,
                                                                                                          train_loss,
                                                                                                          reconstr_ppl,
                                                                                                          adv, adv_accuracy,
                                                                                                          separator))
            if best_train_loss[0] is None or best_train_loss[0] > train_loss:
                best_train_loss = (train_loss, reconstr_ppl, adv)

            #  (2) evaluate on the validation set
            valid_loss, reconstr, adv, adv_accuracy = self.eval(self.validData)
            reconstr_ppl = math.exp(min(reconstr, 100))
            print(
                '{}\nValid loss: {}\nReconstruction ppl: {}\nAdv loss: {}, accuracy: {}\n{}\n'.format(separator,
                                                                                                      valid_loss,
                                                                                                      reconstr_ppl,
                                                                                                      adv, adv_accuracy,
                                                                                                      separator))
            if best_val_loss[0] is None or best_val_loss[0] > valid_loss:
                best_val_loss = (valid_loss, reconstr_ppl, adv)

            if epoch % opt.save_every_epoch == 0:
                model_fname = self.save(epoch, valid_loss)

                # print translation to make it easier to see if the model is good
                if opt.translate_src is not None:
                    separator2 = '.' * 70
                    print('\n' + separator2)
                    for target_style in range(1, 3):
                        print('\nSample translation with style {}:\n'.format(target_style))
                        translate.translate(['-src', opt.translate_src, '-model', model_fname, '-target_style', str(target_style),
                                             '-max_sent_length', '30', '-output', 'stdout', '-remove_bpe', '-diff'])
                    print('\n' + separator2 + '\n')

            batchOrder = None
            iteration = None
            resume = False


        print('\n\nBEST RESULTS:\n')

        print('{}\nTrain loss: {}\nReconstruction ppl: {}\nAdv loss: {}\n{}\n'.format(separator,
                                                                                                      best_train_loss[0],
                                                                                                      best_train_loss[1],
                                                                                                      best_train_loss[2],
                                                                                                      separator))
        print('{}\nValid loss: {}\nReconstruction ppl: {}\nAdv loss: {}\n{}\n'.format(separator,
                                                                                                      best_val_loss[0],
                                                                                                      best_val_loss[1],
                                                                                                      best_val_loss[2],
                                                                                                      separator))



