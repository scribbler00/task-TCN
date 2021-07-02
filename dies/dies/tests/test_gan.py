# from operator import add
# import unittest
# import torch.nn as nn

# from dies.gan import GANLearner
# from dies.utils import set_random_states
# from .synthetic_data import SyntheticData
# from fastai.optimizer import RMSProp


# loss_type = 2
# n_epochs = 20
# lr = 1e-3
# n_samples = 8000
# n_features = 4
# n_targets = 2
# val_perc = 0.2
# batch_size = 64
# structure = [8, 16]
# random_state = 42


# def get_gan_dls(add_dim):
#     sd = SyntheticData(
#         n_samples=n_samples,
#         n_features=n_features,
#         n_targets=n_targets,
#         val_perc=val_perc,
#         batch_size=batch_size,
#         add_dim=add_dim,
#     )
#     dls = sd.get_dls()
#     return dls


# def get_gan_learner(dls, generator_model, critic_model):
#     gan_learn = GANLearner(
#         dls=dls,
#         generator=generator_model,
#         critic=critic_model,
#         loss_type=loss_type,
#         opt_func=RMSProp,
#         clip=0.01,
#         n_gen=1,
#         n_crit=4,
#         tabular=False,
#     )
#     return gan_learn


# class TestGan(unittest.TestCase):
#     def setUp(self):
#         self.dls = get_gan_dls(add_dim=False)
#         bm = BaselineModel(n_features, structure, n_targets)
#         self.generator_model = bm.get_generator_model(nn.Linear)
#         self.critic_model = bm.get_critic_model(nn.Linear)
#         self.learner = get_gan_learner(
#             self.dls, self.generator_model, self.critic_model
#         )
#         set_random_states(random_state)

#     def test_generated_samples(self):
#         self.learner.fit(n_epochs, lr)
#         real_x = self.dls.one_batch()[0].numpy()
#         fake_x = self.learner.model.generate_samples(real_x).detach().numpy()

#         self.assertAlmostEqual(real_x.mean(), fake_x.mean(), delta=1)
#         self.assertAlmostEqual(real_x.var(), fake_x.var(), delta=1)
#         return


# class TestConvGan(unittest.TestCase):
#     def setUp(self):
#         self.dls = get_gan_dls(add_dim=True)
#         bm = BaselineModel(n_features, structure, n_targets)
#         self.generator_model = bm.get_generator_model(nn.ConvTranspose1d)
#         self.critic_model = bm.get_critic_model(nn.Conv1d)
#         self.learner = get_gan_learner(
#             self.dls, self.generator_model, self.critic_model
#         )
#         set_random_states(random_state)

#     def test_generated_samples(self):
#         self.learner.fit(n_epochs, lr)
#         real_x = self.dls.one_batch()[0].numpy()
#         fake_x = self.learner.model.generate_samples(real_x).detach().numpy()

#         self.assertAlmostEqual(real_x.mean(), fake_x.mean(), delta=1)
#         self.assertAlmostEqual(real_x.var(), fake_x.var(), delta=1)
#         return