__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


from torch import nn
import torch
from fastai.layers import *


class VILoss(nn.Module):
    """
    Calculate the Kullback-Leibler divergence loss.
    Parameters
    ----------
    model : dies.embedding
        the given embedding to base the loss on
    lambd : float
        scalar for the loss
    """

    def __init__(
        self,
        model,
        base_loss=torch.nn.MSELoss(),
        kl_weight=1,
    ):
        super(VILoss, self).__init__()
        self.base_loss = base_loss
        self.model = model
        self.kl_weight = kl_weight

    def forward(self, y, y_hat):
        """

        Parameters
        ----------
        y : pytorch.Tensor
            any given tensor. Shape: [n, ]
        y_hat : pytorch.Tensor
            a tensor with the same shape as 'y'

        Returns
        -------
        pytorch.Tensor
            the resulting accumulated loss
        """
        base_loss = self.base_loss(y, y_hat)

        kl = self.model.nn_kl_divergence()
        loss = base_loss + self.kl_weight * kl

        return loss


class CnnMSELoss(torch.nn.MSELoss):
    def __init__(self):
        """
        Calculate the MSELoss and take the mean over all features
        """
        super(CnnMSELoss, self).__init__(None, None, "mean")

    def forward(self, input, target):
        """

        Parameters
        ----------
        input : pytorch.Tensor
            any given tensor. Shape: [n, ]
        target : pytorch.Tensor
            a tensor with the same shape as 'input'

        Returns
        -------
        pytorch.Tensor
            the resulting loss
        """
        return torch.mean(torch.mean(torch.mean(torch.pow((target - input), 2), 2), 0))


class VAEReconstructionLoss(nn.Module):
    def __init__(self, model):
        """
        Calculate the sum of the Kullback–Leibler divergence loss and the loss of any given function
        Parameters
        ----------
        model : dies.autoencoder
            model of the autoencoder for which the loss is to be calculated
        """
        super(VAEReconstructionLoss, self).__init__()
        self.reconstruction_function = torch.nn.MSELoss()
        self.model = model

    def forward(self, x, x_hat):
        """

        Parameters
        ----------
        x : pytorch.Tensor
            any given tensor. Shape: [n, ]
        x_hat : pytorch.Tensor
            a tensor with the same shape as 'x'

        Returns
        -------
        pytorch.Tensor
            the resulting accumulated loss
        """
        recon_x = x_hat

        # how well do input x and output recon_x agree?

        generation_loss = self.reconstruction_function(recon_x, x)
        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        if self.model.embedding_module is None:
            mu, logvar = self.model.get_posteriors(x)
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # note the negative D_{KL} in appendix B of the paper
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # Normalise by same number of elements as in reconstruction
            KLD /= x.shape[0] * x.shape[1]

            # BCE tries to make our reconstruction as accurate as possible
            # KLD tries to push the distributions as close as possible to unit Gaussian
        else:
            KLD = 0
        return generation_loss + KLD


class RSSLoss(nn.Module):
    def __init__(self):
        """
        Calculate the Residual sum of squares loss
        """
        super(RSSLoss, self).__init__()

    def forward(self, y, y_hat):
        """

        Parameters
        ----------
        y : pytorch.Tensor
            any given tensor. Shape: [n, ]
        y_hat : pytorch.Tensor
            a tensor with the same shape as 'y'

        Returns
        -------
        pytorch.Tensor
            the resulting loss
        """
        return ((y - y_hat) ** 2).sum()
