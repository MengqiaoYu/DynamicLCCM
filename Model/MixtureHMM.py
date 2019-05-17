import numpy as np
from scipy.misc import logsumexp
from ChoiceModels import TransitionModel, LogitChoiceModel
import logging
from datetime import datetime

__author__ = "Mengqiao Yu"
__email__ = "mengqiao.yu@berkeley.edu"

logger = logging.getLogger()
logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    filename=datetime.now().strftime('%m-%d_%H_%M_%S') + '.txt')

class BasicHMM():
    """
    class of basic HMM: one choice model, one sequence of data, homogeneous model.
    """
    def __init__(self, num_states=2):
        self.num_states = num_states

    def _forward(self, log_a, log_b, o, log_pi):
        """
        return log of alpha

        Parameters
        ----------
        log_a: ndarray
                log of transition matrix prob
                (K, K)
        log_b: ndarray
                log of emission matrix prob.
                (K, num of alternatives)
        o: ndarray
            one sequence of observations
            (T, 1)
        log_pi: nparray
                log of initial matrix prob
                (K, 1)
        Returns
        -------
        log_alpha: ndarray
                    the probability of seeing the observations y_0, ..., y_t
                    and being in state i at time t, i.e., P(y_0,...,y_t, q_t)
                    (T, K)
        """

        K = log_b.shape[0] # number of states
        T = o.shape[0] # number of timestamps

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
        log of likelihood: log(p(y|theta)) where theta represents all parameters

        Parameters
        ----------
        log_alpha: ndarray
                    P(y_0,...,y_t, q_t)
                    (T, K)
        Returns
        -------
        logliklihood: float
        """
        return logsumexp(log_alpha[-1])

    def _backward(self, log_a, log_b, o):
        """
        return log of beta

        Parameters
        ----------
        log_a: ndarray
                log of transition matrix prob
                (K, K)
        log_b: ndarray
                log of emission matrix prob.
                (K, num of alternatives)
        o: ndarray
            one sequence of observations
            (T, 1)
        Returns
        -------
        log_beta: ndarray
                    p(y_t+1, ..., y_T|q_t)
                    (T, K)

        """
        K = log_b.shape[0] # number of states
        T = o.shape[0] # number of timestamps

        log_beta = np.zeros((T, K))
        # We don't need to specify the log_beta[t-1, :] since we have set to 0.

        for t in range(T-2, -1, -1):
            for i in range(K):
                log_beta[t, i] = logsumexp(log_beta[t+1, :]
                                           + log_a[i, :]
                                           + log_b[:, o[t+1]])
        return log_beta

    def _calc_log_xi(self, log_a, log_b, log_alpha, log_beta, o, log_ll):
        """
        calculate log of xi built on alpha and beta.

        Parameters
        ----------
        log_a: ndarray
                log of transition matrix prob
                (K, K)
        log_b: ndarray
                log of emission matrix prob.
                (K, num of alternatives)
        log_alpha: ndarray
                    P(y_0,...,y_t, q_t)
                    (T, K)
        log_beta: ndarray
                    p(y_t+1, ..., y_T|q_t)
                    (T, K)
        o: ndarray
            one sequence of observations
            (T, 1)
        log_ll: float
        Returns
        -------
        log_xi: ndarray
                xi(q_t, q_t+1) = P(q_t, q_t+1|y)
                (K, K, T)
        """

        K = log_b.shape[0] # number of states
        T = o.shape[0] # number of timestamps

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
        calculate log of gamma

        Parameters
        ----------
        log_alpha: ndarray
        log_beta: ndarray
                    p(y_t+1, ..., y_T|q_t)
                    (T, K)

        Returns
        -------
        log_gamma: ndarray
            gamma(q_t) = P(q_t | y)
            (K,T)
        """
        log_gamma = log_alpha + log_beta - log_ll
        return log_gamma.T

    def _forward_backward(self, log_a, log_b, o, log_pi):
        """
        E step

        Parameters
        ----------
        log_a: ndarray
                log of transition matrix prob
                (K, K)
        log_b: ndarray
                log of emission matrix prob.
                (K, num of alternatives)
        o: ndarray
            one sequence of observations
            (T, 1)
        log_pi: nparray
                log of initial matrix prob

        Returns
        -------
        log_xi: ndarray
                xi(q_t, q_t+1) = P(q_t, q_t+1|y)
                (K, K, T)
        log_gamma: ndarray
                    gamma(q_t) = P(q_t | y)
                    (K, T)
        log_ll: float
        """

        log_alpha = self._forward(log_a, log_b, o, log_pi)

        log_beta = self._backward(log_a, log_b, o)

        log_ll = self._cal_log_likelihood(log_alpha)

        log_xi = self._calc_log_xi(log_a, log_b, log_alpha, log_beta, o, log_ll)

        log_gamma = self._calc_log_gamma(log_alpha, log_beta, log_ll)

        return log_xi, log_gamma, log_ll

    def m_step(self, log_xi, log_gamma, o):
        """
        M step

        Parameters
        ----------
        log_xi: ndarray
                xi(q_t, q_t+1) = P(q_t, q_t+1|y)
                (K, K, T)
        log_gamma: ndarray
                    gamma(q_t) = P(q_t | y)
                    (K, T)
        o: ndarray
            one sequence of observations
            (T, 1)

        Returns
        -------
        log_a_hat: ndarray
                estimated log of transition matrix prob
                (K, K)
        log_b_hat: ndarray
                estimated log of emission matrix prob.
                (K, num of alternatives)
        log_pi_hat: nparray
                estimated log of initial matrix prob
                (K, 1)
        """

        T = o.shape[0] # number of time steps
        C = len(np.unique(o)) # number of alternatives

        log_pi_hat = log_gamma[:, 0]
        log_a_hat = np.zeros((self.num_states, self.num_states))
        log_b_hat = np.zeros((self.num_states, C))

        for i in range(self.num_states):
            for j in range(self.num_states):
                log_a_hat[i, j] = logsumexp(log_xi[i, j, :T-1]) \
                                  - logsumexp(log_gamma[i, :T-1])

            for k in range(C):
                filter_vals = (o == k).nonzero()[0]
                log_b_hat[i, k] = logsumexp(log_gamma[i, filter_vals]) \
                                  - logsumexp(log_gamma[i, :T])

        return log_a_hat, log_b_hat, log_pi_hat

    def initialize(self, o, rand_seed = 0):
        """
        Random sample for initial values of parameters

        Parameters
        ----------
        o: ndarray
            one sequence of observations
            (T, 1)
        rand_seed: int
                    For reproductive results, set rand_seed here.
        Returns
        -------
        log_a: ndarray
                log of transition matrix prob
                (K, K)
        log_b: ndarray
                log of emission matrix prob.
                (K, num of alternatives)
        log_pi: nparray
                log of initial matrix prob
                (K, 1)
        """
        np.random.seed(rand_seed)
        C = len(np.unique(o)) # number of alternatives

        log_pi = np.log(np.ndarray.flatten(np.random.dirichlet(
            np.ones(self.num_states), size=1)))
        log_a = np.log(np.random.dirichlet(np.ones(self.num_states),
                                           size=self.num_states))
        log_b = np.log(np.random.dirichlet(np.ones(C), size=self.num_states))

        return log_a, log_b, log_pi

    def print_results(self, log_a, log_b, log_pi):
        logger.info("\tHere is the initial matrix:")
        logger.info("\t\t", np.exp(log_pi))
        logger.info("\tHere is the transition matrix:")
        logger.info("\t\t", np.exp(log_a))
        logger.info("\tHere is the emission matrix:")
        logger.info("\t\t", np.exp(log_b))

    def train(self, obs, cutoff_value):

        # Initialization
        log_a, log_b, log_pi = self.initialize(o=obs)
        logger.info("The initial values are:")
        self.print_results(log_a=log_a, log_b=log_b, log_pi=log_pi)

        # Start training
        logger.info("Start training:")
        log_xi, log_gamma, log_ll = self._forward_backward(log_a=log_a,
                                                           log_b=log_b,
                                                           o=obs,
                                                           log_pi=log_pi)
        before = log_ll
        increase = cutoff_value + 1
        i = 0
        while(increase <= 0 or increase > cutoff_value):
            i += 1
            log_a, log_b, log_pi = self.m_step(log_xi=log_xi,
                                               log_gamma=log_gamma,
                                               o=obs)
            log_xi, log_gamma, log_ll = self._forward_backward(log_a=log_a,
                                                               log_b=log_b,
                                                               o=obs,
                                                               log_pi=log_pi)
            after = log_ll
            increase = after - before
            before = after

            # Print progress
            logger.info("\tThis is %d iteration, ll = %s." %(i, after))
            if i % 200 == 0:
                self.print_results(log_a=log_a, log_b=log_b, log_pi=log_pi)

        # Print final results
        logger.info("\tThe estimation results are:")
        self.print_results(log_a=log_a, log_b=log_b, log_pi=log_pi)
        logger.info("-----------------------THE END-----------------------")

class MixtureHMM(BasicHMM):
    """
    class of homogeneous Mixture HMM with two added features:
    Feature 1: deal with multiple sequences with same number of time stamps.
    Feature 2: deal with multiple choice models, 
            i.e., multiple choices(observations) at one timestep.
    """
    def _forward(self, log_a, log_choice_prob, log_pi):
        """
        return log of alpha

        Parameters
        ----------
        log_a: ndarray
                log of transition matrix prob
                (K, K)
        log_choice_prob: ndarray
                        log of all emission probs from state i at timestamp t
                        logP(y_1) + logP(y_2)
                        (T, K)
        log_pi: ndarray
                log of initial matrix prob
                (K, 1)
        Returns
        -------
        log_alpha: ndarray
                    P(y_0,...,y_t, q_t)
                    (T, K)
        """

        T = log_choice_prob.shape[0] # number of timestamps
        K = log_choice_prob.shape[1] # number of states

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
        """
        return log of beta

        Parameters
        ----------
        log_a: ndarray
                log of transition matrix prob
                (K, K)
        log_choice_prob: ndarray
                        log of all emission probs from state i at timestamp t
                        logP(y_1) + logP(y_2)
                        (T, K)
        Returns
        -------
        log_beta: ndarray
                    p(y_t+1, ..., y_T|q_t)
                    (T, K)
        """

        T = log_choice_prob.shape[0] # number of timestamps
        K = log_choice_prob.shape[1] # number of states

        log_beta = np.zeros((T, K))

        for t in range(T-2, -1, -1):
            log_beta[t, :] = logsumexp(log_beta[t+1, :]
                                       + log_a
                                       + log_choice_prob[t+1, :], axis = 1)

        return log_beta

    def _forward_backward(self, log_a, log_choice_prob, log_pi):
        """
        E step

        Parameters
        ----------
        log_a: ndarray
                log of transition matrix prob
                (K, K)
        log_choice_prob: ndarray
                        log of all emission probs from state i at timestamp t
                        logP(y_1) + logP(y_2)
                        (T, K)
        log_pi: ndarray
                log of initial matrix prob
        Returns
        -------
        log_xi: ndarray
                xi(q_t, q_t+1) = P(q_t, q_t+1|y)
                (K, K, T)
        log_gamma: ndarray
            gamma(q_t) = P(q_t | y)
            (K, T)
        log_ll: float
        """

        log_alpha = self._forward(log_a, log_choice_prob, log_pi)

        log_beta = self._backward(log_a, log_choice_prob)

        log_ll = self._cal_log_likelihood(log_alpha)

        log_xi = self._calc_log_xi(log_a, log_choice_prob,
                                   log_alpha, log_beta, log_ll)

        log_gamma = self._calc_log_gamma(log_alpha, log_beta, log_ll)

        return log_xi, log_gamma, log_ll

    def _calc_log_xi(self, log_a, log_choice_prob, log_alpha, log_beta, log_ll):
        """
        calculate log of xi built on alpha and beta.

        Parameters
        ----------
        log_a: ndarray
                log of transition matrix prob
                (K, K)
        log_choice_prob: ndarray
                        log of all emission probs from state i at timestamp t
                        logP(y_1) + logP(y_2)
                        (T, K)
        log_alpha: ndarray
                    the probability of seeing the observations y_0, ..., y_t
                    and being in state i at time t.
                    (T, K)
        Returns
        -------
        log_xi: ndarray
                xi(q_t, q_t+1) = P(q_t, q_t+1|y)
                (K, K, T)
        """

        T = log_choice_prob.shape[0] # number of timestamps
        K = log_choice_prob.shape[1] # number of states

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
        calculate log of joint prob of choices for one sequence of observations

        Parameters
        ----------
        o: ndarray
            one sequence of observations
            (T, num_of_choice_models)
        log_b: list of emission matrices with length of num_of_choice_models;
                each matrix: (K,  num of alternatives in each choice model))
        Returns
        -------
        log_prob_choices: ndarray
                            (T * K)
        """

        T = o.shape[0]
        num_of_choice_models = o.shape[1]
        log_choice_prob = np.zeros((T, self.num_states))

        for t in range(T):
            for i in range(self.num_states):
                log_choice_prob[t, i] \
                    = np.sum([log_b[c][i, o[t][c]]
                              for c in range(num_of_choice_models)])

        return log_choice_prob

    def m_step(self):
        """
        M step: re-estimate parameters
        """
        # Be careful with the mean of logsumexp, which is incorrect!
        self.log_pi = np.log(np.mean(
            [np.exp(self.log_gammas[s][:, 0])
             for s in range(self.num_seq)], axis=0))

        for i in range(self.num_states):

            # calculate log_a: transition matrix
            for j in range(self.num_states):
                sum_si = 0
                sum_gamma = 0
                for s in range(self.num_seq):
                    sum_si += np.sum(np.exp(self.log_xis[s][i, j,
                                            :self.num_timesteps-1]))
                    sum_gamma += np.sum(np.exp(self.log_gammas[s][i,
                                               :self.num_timesteps-1]))
                self.log_a[i, j] = np.log(sum_si) - np.log(sum_gamma)

            # calculate log_b: emission matrix for each choice model
            for c in range(self.num_choice_models):
                for k in range(self.num_choices[c]):
                    sum_gamma_y = 0
                    sum_gamma = 0
                    for s in range(self.num_seq):
                        try:
                            sum_gamma_y \
                                += np.sum(np.exp(
                                self.log_gammas[s]
                                [i, (self.obs_seq[s][:, c] == k).nonzero()[0]]))
                        except ValueError:
                            pass
                        sum_gamma \
                            += np.sum(np.exp(self.log_gammas[s]
                                             [i, :self.num_timesteps]))
                    self.log_b[c][i, k] = np.log(sum_gamma_y) \
                                          - np.log(sum_gamma)

    def e_step(self):
        """
        calculate log_xi, log_gamma, log_ll for all sequences.
        """

        self.log_xis = []
        self.log_gammas = []
        self.log_lls = []

        for obs in self.obs_seq:
            log_choice_prob = self.cal_log_prob_choices(log_b=self.log_b, o=obs)
            log_xi, log_gamma, log_ll = \
                self._forward_backward(log_a=self.log_a,
                                       log_choice_prob=log_choice_prob,
                                       log_pi=self.log_pi)
            self.log_xis.append(log_xi)
            self.log_gammas.append(log_gamma)
            self.log_lls.append(log_ll)

    def initialize(self):
        # np.random.seed(rand_seed)

        log_pi = np.log(np.ndarray.flatten(
            np.random.dirichlet(np.ones(self.num_states), size=1)))
        log_a = np.log(
            np.random.dirichlet(np.ones(self.num_states),size=self.num_states))
        log_b = []
        for c in range(self.num_choice_models):
            # c represents one choice model
            log_b_c = np.log(np.random.dirichlet(np.ones(self.num_choices[c]), size=self.num_states))
            log_b.append(log_b_c)
        return log_a, log_b, log_pi

    def train(self, obs_seq, cutoff_value, max_iter):
        """
        train the model.

        Parameters
        ----------
        cutoff_value: float
            one stopping criterion based on the improvement of LL.
        max_iter: int
            another stopping criterion.
        """

        # Initialization
        self.obs_seq = obs_seq
        self.num_seq = len(obs_seq)
        self.num_timesteps = self.obs_seq[0].shape[0]
        self.num_choice_models = self.obs_seq[0].shape[1]
        self.num_choices = [max(len(np.unique(self.obs_seq[i][:, c]))
                                for i in range(self.num_seq))
                            for c in range(self.num_choice_models)]
        self.log_a, self.log_b, self.log_pi = self.initialize()
        logger.info("The initial values are:")
        self.print_results(log_a=self.log_a,
                           log_b=self.log_b,
                           log_pi=self.log_pi)

        #Start training
        logger.info("Start training:")
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
                self.print_results(log_a=self.log_a,
                                   log_b=self.log_b,
                                   log_pi=self.log_pi)

        # Print final results
        logger.info("\tThe estimation results are:")
        self.print_results(log_a=self.log_a,
                           log_b=self.log_b,
                           log_pi=self.log_pi)

        logger.info("-----------------------THE END-----------------------")

class HeteroMixtureHMM(MixtureHMM):
    """
    In addition to the features of MixtureHMM:
    Feature 1: deal with multiple sequences with same number of time stamps.
    Feature 2: deal with multiple choice models, 
            i.e., multiple choices(observations) at one timestep.
    class of heterogeneous Mixture HMM adds two other features:
    Feature 3: heterogeneous HMM: build logit model for transition model.
    Feature 4: calculate standard error and p value for each covariate.
    """
    def __init__(self, num_states):
        self.set_dataframe_flag = False
        super().__init__(num_states)

    def _forward(self, log_trans_prob, log_choice_prob, log_pi):
        """
        forward step in Baum–Welch algorithm.

        Parameters
        ----------
        log_trans_prob: ndarray
            (num of states, T, num of states)
        log_choice_prob: ndarray
            (T, num of states)
        log_pi: ndarray
            log of initial matrix prob (num of states, )

        Returns
        -------
        log_alpha: ndarray
            the probability of seeing the observations y_0, ..., y_t
            and being in state i at time t.
            (T, num of states)
        """
        T = log_choice_prob.shape[0] # number of timestamps
        K = log_choice_prob.shape[1] # number of states
        log_alpha = np.zeros((T, K))

        # alpha(q_0)
        log_alpha[0, :] = log_pi + log_choice_prob[0, :]

        # alpha(q_t)
        for t in range(1, T):
            log_alpha[t, :] = logsumexp(log_alpha[t-1, :]
                                        + log_trans_prob[:, t-1, :].T, axis=1) \
                              + log_choice_prob[t, :]

        return log_alpha

    def _backward(self, log_trans_prob, log_choice_prob):
        """
        backward step in Baum–Welch algorithm.

        Parameters
        ----------
        log_trans_prob: ndarray
            (num of states, T, num of states)
        log_choice_prob: ndarray
            (T, num of states)
        log_pi: ndarray
            log of initial matrix prob (num of states, )

        Returns
        -------
        log_beta: ndarray
             the probability of the ending partial sequence y_t+1, ..., y_T
             given starting state i at time t.
            (T, num of states)
        """
        T = log_choice_prob.shape[0] # number of timestamps
        K = log_choice_prob.shape[1] # number of states

        log_beta = np.zeros((T, K))
        # Note: We don't need to specify the log_beta[T, :]
        # since we have set it to zero.

        for t in range(T-2, -1, -1):
            log_beta[t, :] = logsumexp(log_beta[t+1, :]
                                       + log_trans_prob[:, t, :]
                                       + log_choice_prob[t+1, :], axis = 1)

        return log_beta

    def _calc_log_xi(self, log_trans_prob, log_choice_prob,
                     log_alpha, log_beta, log_ll):
        """
        calculate xi built on alpha and beta.

        Parameters
        ----------
        log_trans_prob: ndarray
            (num of states, T, num of states)
        log_choice_prob: ndarray
            (T, num of states)
        log_alpha: ndarray
            (T, num of states)
        log_beta: ndarray
            (T, num of states)
        log_ll: float

        Returns
        -------
        log_xi: ndarray
            xi(q_t, q_t+1) = P(q_t, q_t+1|y) (num of states, num of states, T)
        """

        T = log_choice_prob.shape[0] # number of timestamps
        K = log_choice_prob.shape[1] # number of states

        log_xi = np.zeros((K, K, T))

        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    log_xi[i, j, t] = log_alpha[t, i] \
                                      + log_choice_prob[t+1, j] \
                                      + log_beta[t+1, j] \
                                      + log_trans_prob[i, t, j] \
                                      - log_ll
        return log_xi

    def _forward_backward(self, log_trans_prob, log_choice_prob, log_pi):
        """
        forward backward algorithm.

        Parameters
        ----------
        log_trans_prob: ndarray
            (num of states, T, num of states)
        log_choice_prob: ndarray
            (T, num of states)
        log_pi: ndarray
            log of initial matrix prob (num of states, )

        Returns
        -------
        log_xi: ndarray
            xi(q_t, q_t+1) = P(q_t, q_t+1|y) (num of states, num of states, T)
        log_gamma: ndarray
            gamma(q_t) = P(q_t | y) (num of states, T)
        log_ll: float
        """

        log_alpha = self._forward(log_trans_prob, log_choice_prob, log_pi)

        log_beta = self._backward(log_trans_prob, log_choice_prob)

        log_ll = self._cal_log_likelihood(log_alpha)

        log_xi = self._calc_log_xi(log_trans_prob, log_choice_prob,
                                   log_alpha, log_beta, log_ll)

        log_gamma = self._calc_log_gamma(log_alpha, log_beta, log_ll)

        return log_xi, log_gamma, log_ll

    def e_step(self):
        """
        calculate log_xi, log_gamma, log_ll (forward backward) for all sequences
        """
        self.log_xis = []
        self.log_gammas = []
        self.log_lls = []

        ### Calculate log_b
        # log_b: (num_of_choice_models, K * num_of_choices in each choice model)
        log_b = []
        for c in range(self.num_choice_models):
            # c represents one choice model
            log_b_c = np.vstack([self.choice_models[i][c].predict_log_proba(1)
                                        for i in range(self.num_states)])
            log_b.append(log_b_c)

        ### Execute forward backward for all sequences.
        for i, obs in enumerate(self.obs_seq):

            ### Calculate the transition log_prob
            # trans_X[i]: (T, num of covariates)
            # log_trans_prob: (num of states, T, num of states)
            log_trans_prob = np.zeros((self.num_states,
                                       self.num_timesteps, self.num_states))
            for i in range(self.num_states):
                log_trans_prob[i, :, :] = \
                    self.trans_models[i].predict_log_proba(self.trans_X[i])

            ### Calculate the emission matrix (log_prob)
            # log_choice_prob: (T, num of states)
            log_choice_prob = self.cal_log_prob_choices(log_b=log_b, o=obs)

            ### forward backward
            log_xi, log_gamma, log_ll = self._forward_backward(
                log_trans_prob=log_trans_prob,
                log_choice_prob=log_choice_prob,
                log_pi=self.log_pi)
            self.log_xis.append(log_xi)
            self.log_gammas.append(log_gamma)
            self.log_lls.append(log_ll)

    def m_step(self):
        """
        re-estimate parameters
        """
        # Note: Be careful with the mean of logsumexp, which is incorrect!
        self.log_pi = np.log(np.mean([np.exp(self.log_gammas[s][:, 0])
                                      for s in range(self.num_seq)], axis=0))

        for i in range(self.num_states):

            # re-estimate transition model
            # y: np (T * self.num_seq, num_states)
            y = np.exp(np.vstack([log_xi[i, :, :].T
                                  for log_xi in self.log_xis]))
            self.trans_models[i].fit(self.trans_X, y)

            # re-estimate choice models
            for c in range(self.num_choice_models):
                # X actually represents constant.
                X = np.ones((self.num_seq * self.num_timesteps, 1))

                # y: np (T * self.num_seq, )
                y = np.hstack([self.obs_seq[i][:, c]
                               for i in range(self.num_seq)])
                assert y.shape == (self.num_timesteps * self.num_seq, ), \
                    "The shape of choice variable is wrong!"

                sample_weight = np.exp(np.hstack(
                    [log_gamma[i, :] for log_gamma in self.log_gammas]))
                self.choice_models[i][c].fit(X, y, sample_weight)

    def initialize(self):
        """
        initialize each parameters.
        """

        # For deterministic result, set rand_seed here. Also for LinearModels.py
        # np.random.seed(rand_seed)

        # initial matrix
        log_pi = np.log(np.ndarray.flatten(
            np.random.dirichlet(np.ones(self.num_states), size=1)))

        # transition model
        trans_models = []
        for i in range(self.num_states):
            trans_model = TransitionModel(
                num_states=self.num_states,
                num_covariates=self.num_trans_covariates
            )
            trans_models.append(trans_model)

        # choice models
        # choice_models is a list of models: (num_states, num_choice_models)
        choice_models = []
        for i in range(self.num_states):
            choice_model = [LogitChoiceModel(num_choices=self.num_choices[c])
                            for c in range(self.num_choice_models)]
            choice_models.append(choice_model)

        return trans_models, choice_models, log_pi

    def set_dataframe(self,
                      samples,
                      header = [],
                      choices = [],
                      trans_cov=[]):
        """
        Extract useful data.

        Parameters
        ----------
        samples: list of ndarray np with length of number of people;
                each np array: (T, num_of_choice_models + num of covariates)

        Returns
        -------
        obs_seq: list of np arrays with length of number of people;
                each np array: (T, num_of_choice_models)
        trans_X: list of np arrays with length of number of people;
                each np array: (T, num of covariates)
        """
        obs_seq = []
        trans_X = []

        logger.info("The covariates are:")
        logger.info(trans_cov)

        for sample in samples:
            obs_seq.append(sample[:, [header.index(name)
                                      for name in choices]].astype(int))
            trans_X.append(sample[:, [header.index(name)
                                      for name in trans_cov]])

        self.obs_seq, self.trans_X = obs_seq, trans_X
        self.set_dataframe_flag = True

    def print_results(self,
                      trans_models,
                      choice_models,
                      log_pi,
                      print_std = False):
        """
        Set print format here.
        
        Parameters
        ----------
        trans_models: a list of logit models with number of states.
        choice_models: a list of list of choice models. (num of states, num of choice models)
        log_pi: ndarray
        print_std: bool
                set to True at the last step to calculate standard error and p value.
        """

        float_formatter = lambda x: "%.3f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})

        logger.info("\tHere is the initial matrix:")
        logger.info("\t\t", np.exp(log_pi), '\n')

        logger.info("\tHere is the transition model:")
        for i in range(self.num_states):
            logger.info("\t\tThis is the transition model for state %d" %(i+1))
            logger.info("\t\t\t", self.trans_models[i].get_params())
            if print_std:
                std, p = self.trans_models[i].get_std()
                logger.info("\t\t\t", p, '\n')

        for c in range(self.num_choice_models):
            logger.info("\tFor choice model %d" %(c+1))
            for i in range(self.num_states):
                logger.info("\t\tHere is estimates for state %d:" %(i+1))
                coef, prob = self.choice_models[i][c].get_params()
                logger.info("\t\t\t", coef)
                logger.info("\t\t\t", prob)
                if print_std:
                    std, p = self.choice_models[i][c].get_std()
                    logger.info("\t\t\t", p, '\n')

    def train(self, cutoff_value, max_iter):
        """
        train the model.

        Parameters
        ----------
        cutoff_value: float
            one stopping criterion based on the improvement of LL.
        max_iter: int
            another stopping criterion.
        """

        assert self.set_dataframe_flag == True, \
            "Run model.set_dataframe before training."

        # Basic information of the model framework
        self.num_seq = len(self.obs_seq)
        self.num_trans_covariates = self.trans_X[0].shape[1]
        self.num_timesteps = self.obs_seq[0].shape[0]
        self.num_choice_models = self.obs_seq[0].shape[1]
        self.num_choices = [max(len(np.unique(self.obs_seq[i][:, c]))
                                for i in range(self.num_seq))
                            for c in range(self.num_choice_models)]

        # Initialization
        self.trans_models, self.choice_models, self.log_pi = self.initialize()
        self.print_results(trans_models=self.trans_models,
                           choice_models=self.choice_models,
                           log_pi=self.log_pi,
                           print_std=False)

        # Start training
        logger.info("Start training:")
        self.e_step()
        before_ll = sum(self.log_lls)
        increase = cutoff_value + 1
        i = 0
        while((increase > cutoff_value or increase <= 0) and i < max_iter):
            i += 1
            # Run EM algorithm
            self.m_step()
            self.e_step()
            after_ll = sum(self.log_lls)
            increase = after_ll - before_ll
            before_ll = after_ll
            # Print progress during estimation
            logger.info("\tThis is %d iteration, ll = %s." %(i, after_ll))

        # Print final estimation results.
        logger.info("The estimation results are:")
        self.print_results(trans_models=self.trans_models,
                           choice_models=self.choice_models,
                           log_pi=self.log_pi,
                           print_std=True)
       

        logger.info("-----------------------THE END-----------------------")

