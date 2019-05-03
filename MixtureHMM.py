import numpy as np
from scipy.misc import logsumexp
import logging

__author__ = "Mengqiao Yu"
__email__ = "mengqiao.yu@berkeley.edu"

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

class BasicHMM():

    def __init__(self, num_states=2):
        self.num_states = num_states

    def _forward(self, log_a, log_b, o, log_pi):
        """
        return log_alpha
        :param log_a: log of transition matrix prob
        :type log_a: numpy array
        :param log_b: log of emission matrix prob
        :type log_b: numpy array
        :param log_pi: log of initial matrix prob
        :param o: one sequence of observations
        :type o: numpy array
        """
        K = np.shape(log_b)[0] # number of states
        T = np.shape(o)[0] # number of timestamps

        log_alpha = np.zeros((T, K))

        # alpha(q_0)
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
        return log_beta

    def _calc_log_xi(self, log_a, log_b, log_alpha, log_beta, o, log_ll):
        """
        calculate xi built on alpha and beta.
        xi(q_t, q_t+1) = P(q_t, q_t+1|y)
        return log_xi (K * K * T)
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
                                      + log_a[i, j] \
                                      - log_ll

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

        log_alpha = self._forward(log_a, log_b, o, log_pi)

        log_beta = self._backward(log_a, log_b, o)

        log_ll = self._cal_log_likelihood(log_alpha)

        log_si = self._calc_log_xi(log_a, log_b, log_alpha, log_beta, o, log_ll)

        log_gamma = self._calc_log_gamma(log_alpha, log_beta, log_ll)

        return log_si, log_gamma, log_ll

    def m_step(self, log_si, log_gamma, o):

        T = np.shape(o)[0] # number of timestamps
        C = len(np.unique(o)) # number of choices

        log_pi_hat = log_gamma[:, 0]
        log_a_hat = np.zeros((self.num_states, self.num_states))
        log_b_hat = np.zeros((self.num_states, C))

        for i in range(self.num_states):
            for j in range(self.num_states):
                log_a_hat[i, j] = logsumexp(log_si[i, j, :T-1]) - logsumexp(log_gamma[i, :T-1])

            for k in range(C):
                filter_vals = (o == k).nonzero()[0]
                log_b_hat[i, k] = logsumexp(log_gamma[i, filter_vals]) - logsumexp(log_gamma[i, :T])

        return log_a_hat, log_b_hat, log_pi_hat

    def set_inputs(self, o, rand_seed = 0):
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
            log_a, log_b, log_pi = self.m_step(log_si=log_si, log_gamma=log_gamma, o=obs)
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
    """
    Feature 1: deal with multiple sequences with same number of time stamps.
    Feature 2: deal with multiple choice models.
    Feature 3; heterogeneous HMM
    """
    #TODO: transition matrix adds covariates
    #TODO: add std

    def _forward(self, log_a, log_choice_prob, log_pi):
        """
        return log_alpha
        :param log_a: log of transition matrix prob (K * K)
        :type log_a: numpy array
        :param log_choice_prob: log of all emission probs from state i at timestamp t
                                logP(y_1) + logP(y_2) (T * K)
        :type log_choice_prob: numpy array
        :param log_pi: log of initial matrix prob
        """
        T = np.shape(log_choice_prob)[0] # number of timestamps
        K = np.shape(log_choice_prob)[1] # number of states

        log_alpha = np.zeros((T, K))

        # alpha(q_0)
        log_alpha[0, :] = log_pi + log_choice_prob[0, :]

        # alpha(q_t)
        for t in range(1, T):
            log_alpha[t, :] = logsumexp(log_alpha[t-1, :] + log_a.T, axis=1) \
                              + log_choice_prob[t, :]
        # Here is the detailed version, slower.
        # for t in range(1, T):
        #     for i in range(K):
        #         log_alpha[t, i] = logsumexp(log_alpha[t-1, :] + log_a[:, i]) \
        #                           + log_choice_prob[t, i]

        return log_alpha

    def _backward(self, log_a, log_choice_prob):

        T = np.shape(log_choice_prob)[0] # number of timestamps
        K = np.shape(log_choice_prob)[1] # number of states

        log_beta = np.zeros((T, K))
        # We don't need to specify the log_beta[t-1, :] since we have set to zero.

        for t in range(T-2, -1, -1):
            log_beta[t, :] = logsumexp(log_beta[t+1, :]
                                       + log_a
                                       + log_choice_prob[t+1, :], axis = 1)

        return log_beta

    def _forward_backward(self, log_a, log_choice_prob, log_pi):
        """
        E step
        """

        # Clear check with benchmark
        log_alpha = self._forward(log_a, log_choice_prob, log_pi)
        # print("This is alpha %s" %np.exp(log_alpha))

        # Clear check with benchmark
        log_beta = self._backward(log_a, log_choice_prob)
        # print("This is beta %s" %np.exp(log_beta))

        # Clear check with benchmark
        log_ll = self._cal_log_likelihood(log_alpha)
        # print("This is ll %s." %log_ll)

        # Clear check with benchmark
        log_si = self._calc_log_xi(log_a, log_choice_prob, log_alpha, log_beta, log_ll)
        # for t in range(self.num_timesteps):
        #     print("This is si %s" %np.exp(log_si[:, :, t]))

        # Clear check with benchmark
        log_gamma = self._calc_log_gamma(log_alpha, log_beta, log_ll)
        # print("This is gamma %s" %np.exp(log_gamma[-2:]))

        return log_si, log_gamma, log_ll

    def _calc_log_xi(self, log_a, log_choice_prob, log_alpha, log_beta, log_ll):
        """
        calculate xi built on alpha and beta.
        xi(q_t, q_t+1) = P(q_t, q_t+1|y)
        """

        T = np.shape(log_choice_prob)[0] # number of timestamps
        K = np.shape(log_choice_prob)[1] # number of states

        log_xi = np.zeros((K, K, T))

        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    log_xi[i, j, t] = log_alpha[t, i] \
                                      + log_choice_prob[t+1, j] \
                                      + log_beta[t+1, j] \
                                      + log_a[i, j] \
                                      - log_ll
        return log_xi

    def cal_log_prob_choices(self, log_b, o):
        """
        return log_prob_choices: (T * K)
        :param o: (T * num_of_choice_models)
        :param log_b: (num_of_choice_models * (K * num_of_choices in each choice model))
        """
        T = np.shape(o)[0]
        num_of_choice_models = np.shape(o)[1]
        log_choice_prob = np.zeros((T, self.num_states))
        for t in range(T):
            for i in range(self.num_states):
                log_choice_prob[t, i] = np.sum([log_b[c][i, o[t][c]] for c in range(num_of_choice_models)])
        return log_choice_prob

    def m_step(self):
        """
        calculate estimated parameters
        """
        # Be careful with the mean of logsumexp, which is incorrect!
        self.log_pi = np.log(np.mean([np.exp(self.log_gammas[s][:, 0]) for s in range(self.num_seq)], axis=0))

        for i in range(self.num_states):

            # calculate log_a: transition matrix
            for j in range(self.num_states):
                sum_si = 0
                sum_gamma = 0
                for s in range(self.num_seq):
                    sum_si += np.sum(np.exp(self.log_sis[s][i, j, :self.num_timesteps-1]))
                    sum_gamma += np.sum(np.exp(self.log_gammas[s][i, :self.num_timesteps-1]))
                self.log_a[i, j] = np.log(sum_si) - np.log(sum_gamma)

            # calculate log_b: emission matrix for each choice model
            for c in range(self.num_choice_models):
                for k in range(self.num_choices[c]):
                    sum_gamma_y = 0
                    sum_gamma = 0
                    for s in range(self.num_seq):
                        try:
                            sum_gamma_y += np.sum(np.exp(self.log_gammas[s][i, (self.obs_seq[s][:, c] == k).nonzero()[0]]))
                        except ValueError:
                            pass
                        sum_gamma += np.sum(np.exp(self.log_gammas[s][i, :self.num_timesteps]))
                    self.log_b[c][i, k] = np.log(sum_gamma_y) - np.log(sum_gamma)

    def e_step(self):
        """calculate log_si, log_gamma, log_ll for all sequences"""
        self.log_sis = []
        self.log_gammas = []
        self.log_lls = []
        for obs in self.obs_seq:
            self.log_choice_prob = self.cal_log_prob_choices(log_b=self.log_b, o=obs)
            log_si, log_gamma, log_ll = self._forward_backward(log_a=self.log_a,
                                                              log_choice_prob=self.log_choice_prob,
                                                              log_pi=self.log_pi)
            self.log_sis.append(log_si)
            self.log_gammas.append(log_gamma)
            self.log_lls.append(log_ll)

    def initialize(self):
        # np.random.seed(rand_seed)

        log_pi = np.log(np.ndarray.flatten(np.random.dirichlet(np.ones(self.num_states), size=1)))
        log_a = np.log(np.random.dirichlet(np.ones(self.num_states),size=self.num_states))
        log_b = []
        for c in range(self.num_choice_models):
            # c represents one choice model
            log_b_c = np.log(np.random.dirichlet(np.ones(self.num_choices[c]), size=self.num_states))
            log_b.append(log_b_c)
        return log_a, log_b, log_pi

    def train(self, obs_seq, cutoff_value, max_iter):

        # Initialization
        self.obs_seq = obs_seq
        self.num_seq = len(obs_seq)
        self.num_timesteps = np.shape(self.obs_seq[0])[0]
        self.num_choice_models = np.shape(self.obs_seq[0])[1]
        self.num_choices = [max(len(np.unique(self.obs_seq[i][:, c])) for i in range(self.num_seq)) for c in range(self.num_choice_models)]
        self.log_a, self.log_b, self.log_pi = self.initialize()
        logger.info("The initial values are:")
        self.print_results(log_a=self.log_a, log_b=self.log_b, log_pi=self.log_pi)

        #Start training
        self.e_step()
        before_ll = sum(self.log_lls)

        increase = cutoff_value + 1
        i = 0
        while(increase <= 0 or increase > cutoff_value or i < max_iter):
            i += 1

            # Run EM algorithm
            self.m_step()
            self.e_step()
            after_ll = sum(self.log_lls)
            increase = after_ll - before_ll
            before_ll = after_ll

            # Print progress during estimation
            if i % 10 == 1:
                logger.info("\tThis is %d iteration, ll = %s." %(i, after_ll))
                self.print_results(log_a=self.log_a, log_b=self.log_b, log_pi=self.log_pi)
                # import pdb;pdb.set_trace()

        # Print final results
        logger.info("\tThe estimation results are:")
        self.print_results(log_a=self.log_a, log_b=self.log_b, log_pi=self.log_pi)

        logger.info("-----------------------THE END-----------------------")
