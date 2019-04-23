import numpy as np
from scipy.misc import logsumexp
# import warnings
import logging

__author__ = "Mengqiao Yu"
__email__ = "mengqiao.yu@berkeley.edu"

# warnings.simplefilter("ignore")
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

class BasicHMM():

    def __init__(self, num_states=2):
        self.num_states = num_states

    def _forward(self, log_a, log_b, o, log_pi):
        """
        return log_alpha
        :param o: a sequence of observations
        :type o: numpy array
        """
        K = np.shape(log_b)[0] # number of states
        T = np.shape(o)[0] # number of timestamps

        log_alpha = np.zeros((T, K))

        # alpha(q_0)
        # import pdb; pdb.set_trace()
        log_alpha[0, :] = log_pi + log_b[:, o[0]]

        # alpha(q_t)
        for t in range(1, T):
            for i in range(K):
                log_alpha[t, i] = logsumexp(log_alpha[t-1, :] + log_a[:, i]) \
                                  + log_b[i, o[t]]

        return log_alpha

    def _cal_log_likelihood(self, log_alpha):
        """
        log of P(y)
        """
        # print(logsumexp(log_alpha[-1]))
        return logsumexp(log_alpha[-1])

    def _backward(self, log_a, log_b, o):

        K = np.shape(log_b)[0] # number of states
        T = np.shape(o)[0] # number of timestamps

        log_beta = np.zeros((T, K))
        # We don't need to specify the log_beta[t-1, :] since we have set to zero.

        for t in range(T-2, -1, -1):
            for i in range(K):


                log_beta[t, i] = logsumexp(log_beta[t+1, :]
                                           + log_a[i, :]
                                           + log_b[:, o[t+1]])
                # import pdb; pdb.set_trace()
                # if log_beta[t, i] > 0:
                #     logger.debug(t)
                #     logger.debug(i)
                #     logger.debug(log_beta[t+1, :])
                #     logger.debug(log_a[:, i])
                #     logger.debug(log_b[:, o[t+1]])
                # import pdb; pdb.set_trace()
        return log_beta

    def _calc_log_xi(self, log_a, log_b, log_alpha, log_beta, o, log_ll):
        """
        calculate xi built on alpha and beta.
        xi(q_t, q_t+1) = P(q_t, q_t+1|y)
        """

        K = np.shape(log_b)[0] # number of states
        T = np.shape(o)[0] # number of timestamps

        log_xi = np.zeros((K, K, T))

        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    log_xi[i, j, t] = log_alpha[t, i] \
                                      + log_b[j, o[t+1]] \
                                      + log_beta[t+1, j] \
                                      + log_a[i, j]
            log_xi[:, :, t] -= log_ll

        return log_xi

    def _calc_log_gamma(self, log_alpha, log_beta, log_ll):
        """
        gamma(q_t) = P(q_t | y)
        return gamma (K*T)
        """

        log_gamma = log_alpha + log_beta - log_ll
        return log_gamma.T

    def _forward_backward(self, log_a, log_b, o, log_pi):
        """
        E step
        """
        # import pdb; pdb.set_trace()

        log_alpha = self._forward(log_a, log_b, o, log_pi)
        # print("This is log_alpha %s" %log_alpha[-2:])
        # import pdb; pdb.set_trace()

        log_beta = self._backward(log_a, log_b, o)
        # print("This is log_beta %s" %np.exp(log_beta[-2:]))

        log_ll = self._cal_log_likelihood(log_alpha)
        # print("This is ll %s." %log_ll)

        log_si = self._calc_log_xi(log_a, log_b, log_alpha, log_beta, o, log_ll)
        # print("This is log_si %s" %np.exp(log_si[:, :, -1]))

        log_gamma = self._calc_log_gamma(log_alpha, log_beta, log_ll)
        # print("This is log_gamma %s" %np.exp(log_gamma[-1]))

        return log_si, log_gamma, log_ll

    def m_step(self, log_si, log_gamma, o):

        T = np.shape(o)[0] # number of timestamps
        C = len(np.unique(o)) # number of choices

        log_pi_hat = log_gamma[:, 0]
        log_a_hat = np.zeros((self.num_states, self.num_states))
        log_b_hat = np.zeros((self.num_states, C))

        for i in range(self.num_states):
            for j in range(self.num_states):
                # import pdb; pdb.set_trace()
                log_a_hat[i, j] = logsumexp(log_si[i, j, :T-1]) - logsumexp(log_gamma[i, :T-1])

            # import pdb; pdb.set_trace()
            for k in range(C):
                filter_vals = (o == k).nonzero()[0]
                log_b_hat[i, k] = logsumexp(log_gamma[i, filter_vals]) - logsumexp(log_gamma[i, :T])

        return log_a_hat, log_b_hat, log_pi_hat

    def set_inputs(self, o, rand_seed = 10):
        T = np.shape(o)[0] # number of timestamps
        C = len(np.unique(o)) # number of choices

        np.random.seed(rand_seed)
        log_pi = np.log(np.ndarray.flatten(np.random.dirichlet(np.ones(self.num_states), size=1)))
        log_a = np.log(np.random.dirichlet(np.ones(self.num_states),size=self.num_states))
        log_b = np.log(np.random.dirichlet(np.ones(C), size=self.num_states))
        return log_a, log_b, log_pi

    def print_results(self, log_a, log_b, log_pi):
        logger.info("\tHere is the initial matrix:")
        logger.info(np.exp(log_pi))
        logger.info("\tHere is the transition matrix:")
        logger.info(np.exp(log_a))
        logger.info("\tHere is the emission matrix:")
        logger.info(np.exp(log_b))

    def train(self, obs, cutoff_value):

        # Initialization
        log_a, log_b, log_pi = self.set_inputs(o=obs)
        logger.info("The initial values are:")
        self.print_results(log_a=log_a, log_b=log_b, log_pi=log_pi)

        # Start training
        log_si, log_gamma, log_ll = self._forward_backward(log_a=log_a, log_b=log_b, o=obs, log_pi=log_pi)

        before = log_ll
        increase = cutoff_value + 1
        i = 0
        while(increase <= 0 or increase > cutoff_value):
            i += 1
            log_a, log_b, log_pi = self.M_step(log_si=log_si, log_gamma=log_gamma, o=obs)
            log_si, log_gamma, log_ll = self._forward_backward(log_a=log_a, log_b=log_b, o=obs, log_pi=log_pi)
            after = log_ll
            increase = after - before
            before = after

            # Print progress
            if i % 200 == 1:
                logger.info("\tThis is %d iteration, ll = %s." %(i, after))
                self.print_results(log_a=log_a, log_b=log_b, log_pi=log_pi)

        # Print final results
        logger.info("\tThe estimation results are:")
        self.print_results(log_a=log_a, log_b=log_b, log_pi=log_pi)
        logger.info("-----------------------THE END-----------------------")

class MixtureHMM(BasicHMM):
    #TODO: two kinds of choices
    #TODO: transition matrix adds covariates
    #TODO: add std

    def m_step(self):
        """calculate estimated parameters"""
        self.log_pi = np.mean([self.log_gammas[s][:, 0] for s in range(self.num_seq)], axis=0)

        for i in range(self.num_states):
            # calculate log_a: transition matrix
            for j in range(self.num_states):
                sum_si = 0
                sum_gamma = 0
                for s in range(self.num_seq):
                    sum_si += np.sum(np.exp(self.log_sis[s][i, j, :self.num_timesteps-1]))
                    sum_gamma += np.sum(np.exp(self.log_gammas[s][i, :self.num_timesteps-1]))
                self.log_a[i, j] = np.log(sum_si) - np.log(sum_gamma)

            # calculate log_b: emission matrix
            for k in range(self.num_choices):
                # filter_vals = (self.obs_seq[s] == k).nonzero()[0]
                sum_gamma_y = 0
                sum_gamma = 0
                for s in range(self.num_seq):
                    try:
                        sum_gamma_y += np.sum(np.exp(self.log_gammas[s][i, (self.obs_seq[s] == k).nonzero()[0]]))
                    except ValueError:
                        pass
                    sum_gamma += np.sum(np.exp(self.log_gammas[s][i, :self.num_timesteps]))
                self.log_b[i, k] = np.log(sum_gamma_y) - np.log(sum_gamma)

    def e_step(self):
        """calculate log_si, log_gamma, log_ll for all sequences"""
        self.log_sis = []
        self.log_gammas = []
        self.log_lls = []
        for obs in self.obs_seq:
            log_si, log_gamma, log_ll = self._forward_backward(log_a=self.log_a,
                                                              log_b=self.log_b,
                                                              o=obs,
                                                              log_pi=self.log_pi)
            self.log_sis.append(log_si)
            self.log_gammas.append(log_gamma)
            self.log_lls.append(log_ll)

    def train(self, obs_seq, cutoff_value):

        # Initialization
        self.log_a, self.log_b, self.log_pi = self.set_inputs(o=obs_seq[0])
        self.obs_seq = obs_seq
        self.num_seq = len(obs_seq)
        self.num_timesteps = np.shape(self.obs_seq[0])[0]
        self.num_choices = max(len(np.unique(self.obs_seq[i])) for i in range(self.num_seq))
        logger.info("The initial values are:")
        self.print_results(log_a=self.log_a, log_b=self.log_b, log_pi=self.log_pi)

        # Start training
        self.e_step()
        before_ll = sum(self.log_lls)

        increase = cutoff_value + 1
        i = 0
        while(increase <= 0 or increase > cutoff_value):
            i += 1

            # Execute EM algorithm
            self.m_step()
            self.e_step()
            after_ll = sum(self.log_lls)
            increase = after_ll - before_ll
            before_ll = after_ll

            # Print progress during estimation
            if i % 10 == 1:
                logger.info("\tThis is %d iteration, ll = %s." %(i, after_ll))
                self.print_results(log_a=self.log_a, log_b=self.log_b, log_pi=self.log_pi)

        # Print final results
        logger.info("\tThe estimation results are:")
        self.print_results(log_a=self.log_a, log_b=self.log_b, log_pi=self.log_pi)
        logger.info("-----------------------THE END-----------------------")






