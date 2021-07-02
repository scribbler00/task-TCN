__author__ = "Maik Jessulat"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


import torch
import torch.nn as nn
from enum import Enum

from fastai.basics import L
from fastai.learner import *
from fastai.callback.all import *
from fastai.optimizer import *
from fastai.torch_core import to_np


class GANLossType(Enum):
    Basic = 1
    WGan = 2


class GANModule(nn.Module):
    def __init__(
        self,
        generator,
        critic,
        noise_size=100,
        gen_mode=True,
        loss_type=GANLossType.Basic,
    ):
        super(GANModule, self).__init__()
        self.generator = generator
        self.critic = critic
        self.noise_size = noise_size
        self.gen_mode = gen_mode
        self.loss_type = loss_type

    def input_samples(self, x):
        if len(x.shape) == 2:
            return torch.rand(x.shape[0], self.noise_size)
        elif len(x.shape) == 3:
            return torch.rand(x.shape[0], self.noise_size, x.shape[2])

    def generate_samples(self, x):
        samples = self.generator(self.input_samples(x))
        return samples

    def criticise_samples(self, x):
        return self.critic(x)

    def switch(self, gen_mode):
        self.gen_mode = gen_mode

    def forward(self, x_real):
        y_real = self.criticise_samples(x_real)
        x_fake = self.generate_samples(x_real)
        y_fake = self.criticise_samples(x_fake)
        y = (y_real, y_fake)
        return y


class TabularGANModule(GANModule):
    def __init__(
        self,
        generator,
        critic,
        noise_size=100,
        gen_mode=True,
        loss_type=GANLossType.Basic,
    ):
        super(TabularGANModule, self).__init__(generator, critic)
        self.generator = generator
        self.critic = critic
        self.noise_size = noise_size
        self.gen_mode = gen_mode
        self.loss_type = loss_type

    def generate_samples(self, x_cat, x_cont):
        samples = self.generator(self.input_samples(x_cat), self.input_samples(x_cont))
        return samples

    def criticise_samples(self, x_cat, x_cont):
        return self.critic(x_cat, x_cont)

    def forward(self, x_cat, x_cont):
        y_real = self.criticise_samples(x_cat, x_cont)
        x_fake = self.generate_samples(x_cat, x_cont)
        y_fake = self.criticise_samples(x_cat, x_fake)
        return (y_real, y_fake)


class GANLoss(nn.Module):
    def __init__(self, gan_model):
        super(GANLoss, self).__init__()
        self.model = gan_model

    def bce_loss(self, y, t):
        sigmoid = nn.Sigmoid()
        bce_loss = nn.BCELoss()
        loss = bce_loss(sigmoid(y), t)
        return loss

    def forward(self, y, t=None):
        y_real = y[0]
        y_fake = y[1]
        if GANLossType(self.model.loss_type) == GANLossType.WGan:
            if self.model.gen_mode:
                return -torch.abs(torch.mean(y_fake))
            else:
                return -torch.abs(torch.mean(y_real) + torch.mean(y_fake)) / 2
        else:
            if self.model.gen_mode:
                t_fake = torch.ones(y_fake.shape)
                loss = self.bce_loss(y_fake, t_fake)
                return loss
            else:
                t_real = torch.ones(y_real.shape)
                t_fake = torch.zeros(y_fake.shape)
                loss = (
                    self.bce_loss(y_real, t_real) + self.bce_loss(y_fake, t_fake)
                ) / 2
                return loss


class GANTrainer(Callback):
    run_after = TrainEvalCallback

    def __init__(self, clip, n_gen, n_crit):
        super(GANTrainer, self).__init__()
        self.clip = clip
        self.n_gen = n_gen
        self.n_crit = n_crit

    def _set_trainable(self, train_gen, train_crit):
        for p in self.learn.model.generator.parameters():
            p.requires_grad_(train_gen)
        for p in self.learn.model.critic.parameters():
            p.requires_grad_(train_crit)

    def before_train(self):
        self.c_gen = 0
        self.c_crit = 0
        self.learn.model.switch(gen_mode=True)

    def before_batch(self):
        if self.training and self.clip is not None:
            for p in self.learn.model.critic.parameters():
                p.data.clamp_(-self.clip, self.clip)
        if self.learn.model.gen_mode:
            self._set_trainable(train_gen=True, train_crit=False)
        else:
            self._set_trainable(train_gen=False, train_crit=True)

    def after_batch(self):
        if self.training:
            if self.learn.model.gen_mode:
                self.c_gen += 1
                if self.c_gen >= self.n_gen:
                    self.learn.model.switch(gen_mode=False)
                    self.c_gen = 0
            else:
                self.c_crit += 1
                if self.c_crit >= self.n_crit:
                    self.learn.model.switch(gen_mode=True)
                    self.c_crit = 0

    def before_validate(self):
        self._set_trainable(train_gen=False, train_crit=False)

    def after_fit(self):
        self._set_trainable(train_gen=True, train_crit=True)
        self.learn.model.switch(gen_mode=True)


class GANMetric(Metric):
    def __init__(self):
        self.d_mean = None
        self.d_var = None

    def reset(self):
        pass

    def accumulate(self, learn):
        data = learn.dls.valid.one_batch()
        cat_data = data[0]
        cont_data = data[1]
        gen_data = learn.model.generate_samples(cat_data, cont_data)
        self.d_mean = to_np(gen_data.mean()).mean()
        self.d_var = to_np(gen_data.var()).mean()
        # self.d_mean = to_np(torch.abs(gen_data.mean() - cont_data.mean())).mean()
        # self.d_var = to_np(torch.abs((gen_data.var() - cont_data.var()))).mean()
        return None

    @property
    def value(self):
        return [f"{self.d_mean:.2f}", f"{self.d_var:.2f}"]

    @property
    def name(self):
        return ["mean", "var"]


class GANLearner(Learner):
    def __init__(
        self,
        dls,
        generator,
        critic,
        loss_type=GANLossType.Basic,
        cbs=None,
        tabular=False,
    ):
        if loss_type == GANLossType.Basic:
            opt_func = Adam
            clip = None
            n_gen = 1
            n_crit = 1

        elif loss_type == GANLossType.WGan:
            opt_func = RMSProp
            clip = 0.01
            n_gen = 1
            n_crit = 4

        else:
            raise NotImplementedError

        if tabular:
            gan = TabularGANModule(
                generator=generator, critic=critic, loss_type=loss_type
            )
        else:
            gan = GANModule(generator=generator, critic=critic, loss_type=loss_type)

        loss_func = GANLoss(gan)
        trainer = GANTrainer(clip, n_gen=n_gen, n_crit=n_crit)
        cbs = L(trainer)
        metrics = GANMetric()
        super(GANLearner, self).__init__(
            dls, gan, loss_func=loss_func, opt_func=opt_func, cbs=cbs, metrics=metrics
        )