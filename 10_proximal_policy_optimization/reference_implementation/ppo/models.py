from math import pi, sqrt

from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Dense
from keras.models import Input, Model, Sequential
from keras.optimizers import Adam
import numpy as np
from scipy.stats import norm

def get_loss(advantages, actions, var, loss_clipping_epsilon):
    """Return the L_CLIP loss defined in the original PPO paper (2017)

    Arguments:

    advantages-- Finite horizon advantages, passed as an input layer to the
                 actor model. Therefore, it is a Keras tensor.
    action -- The action chosen by the agent, passed as an input layer to
              the actor model. Therefore, it is a Keras tensor.
    var -- Variance. This actor assumes spherical covariance, where the variance
           value isn't learned, but is rather a hyperparameter
    loss_clipping_epsilon -- surrogate loss is clipped in the range between
                             1 - loss_clipping_epsilon and
                             1 + loss_clipping_epsilon


    Notes:
    The L_CLIP loss defined in the original PPO paper (2017) is not just
    a function of y_true (old means of the multivariate gaussian) and y_pred
    (new means of the multivariate gaussian), which are the only two
    parameters accepted by a Keras loss function. The loss involves, in
    addition, the actions chosen by the agent for a given observation
    (required in computing surrogate loss) and the advantages.

    To pass actions and advantages as arguments, we need to use a trick. The
    trick is to define a function that accepts additional inputs and returns
    another function which takes only y_true and y_pred as input.

    def fn_accepting_additional_arguments(additional_argument):

        def loss(y_true, y_pred):
            # additional_argument is now available inside this function
            return value

        return loss

    If the additional argument is not a constant, then define an input
    layer that will receive the argument. The input layer can be passed to
    the function accepting additional arguments. This is exactly what we
    are doing to pass the advantages and actions to the loss function.

    This is a Keras specific trick.
    """

    # Flatten advantages
    advantages = K.squeeze(advantages, axis = -1)

    def loss(y_true, y_pred):

        # Gaussian normalization factor
        normalization_factor = 1/sqrt(2*pi*var)

        # Calculate old pdf
        old_exponent = -K.square(actions - y_true)/(2.*var)
        old_pdf = K.prod(
            normalization_factor*K.exp(old_exponent), axis = -1
            )

        # Calculate new pdf
        exponent = -K.square(actions - y_pred)/(2.*var)
        pdf = K.prod(normalization_factor*K.exp(exponent), axis = -1)

        # Calculate ratio and clipped ratio
        pdf_ratio = pdf/old_pdf
        clipped_pdf_ratio = K.clip(
            pdf_ratio,
            1 - loss_clipping_epsilon,
            1 + loss_clipping_epsilon
            )

        # Get clipped loss
        # Notice the minus sign. Since we want to maximize the
        # objective, we need to minimize the negative objective.
        clipped_loss = -K.minimum(
            pdf_ratio*advantages,
            clipped_pdf_ratio*advantages
            )

        return clipped_loss

    # Return a function which takes only y_true and y_pred as input
    return loss

class DefaultActor(object):
    """The default actor class for this reference implementation.

    You can always write your own implementation of the actor model that
    differs from the default implementation. Just override the methods
    of this class, while keeping the methods signatures same. Then import your
    new class and use it in run_ppo_agent.py instead of this default actor.
    """

    def __init__(self, env, var, lr, loss_clipping_epsilon,
                 model = None, training_logs_file_path = None,
                 model_saving_path = None, model_saving_interval = 1
                 ):
        """
        Arguments:
        env -- A Gym environment (can be vanilla or wrapped). We need this
               for infering input and output dimensions of the model.
        var -- Variance. We are assuming a multivariate Gaussian policy with
               constant spherical covariance. The model outputs means of this
               distribution, one for each action.  The variance value isn't
               learned, but is rather a hyperparameter to be provided by the
               user.
        lr -- Learning rate
        loss_clipping_epsilon -- surrogate loss is clipped in the range between
                                 1 - loss_clipping_epsilon and
                                 1 + loss_clipping_epsilon
        model -- A compiled Keras model. If None, then the model is created.
        training_logs_file_path -- The file where we should save the logs for
                                   training. This is useful for monitoring loss etc.
        model_saving_path -- The model will be saved to this filepath after a
                             certain number of epochs of training
        model_saving_interval -- The model will be saved to model_saving_path after
                                 this many epochs

        Notes about the model:
        We will use a MLP (Multi Layer Perceptron) with two hidden layers
        (with 64 units each) as the actor model in our reference implementation.
        We will use tanh activation for the hidden layer.

        The model assumes that we are dealing with continuous environments
        like BipedalWalker-v2 or Hopper-v1, where the observation shape is (n,),
        where n can be any any integer. The actor outputs means for a
        multivariate gaussian policy (one gaussian for each action) with a
        constant spherical covariance.

        For better performance, you may also want to learn the covariance. In
        the case of diagonal covariance, the actor model should not only output
        means, but also the variances. The loss also needs to be defined
        accordingly.

        If you are dealing with a discrete environment, the actor model will
        directly output probabilities. Therefore, the loss needs to be defined
        differently.

        If learning directly from pixels, the actor model needs to be a CNN
        (Convolutional Neural Network) to be able to learn properly.
        """

        self.env = env
        self.var = var
        self.lr = lr
        self.loss_clipping_epsilon = loss_clipping_epsilon
        self.training_logs_file_path = training_logs_file_path
        self.model_saving_path = model_saving_path
        self.model_saving_interval = model_saving_interval

        # Define the model
        if model is not None:
            self.model = model
        else:
            # If no model is specifiec, create a new model
            observations = Input(shape = self.env.observation_space.shape)
            first_hidden_layer = Dense(64, activation = "tanh")(observations)
            second_hidden_layer = Dense(64, activation = "tanh")(
                first_hidden_layer
                )
            gaussian_means = Dense(self.env.action_space.shape[0])(
                second_hidden_layer
                )

            # The following input layers have no effect on the output. We
            # include them because we need the advantages and the actions for
            # calculating the PPO loss, and having them as inputs is the only
            # way of passing them to the loss function in Keras.
            advantages = Input(shape = (1,))
            actions = Input(shape = self.env.action_space.shape)

            self.model = Model(inputs = [observations, advantages, actions],
                               outputs = gaussian_means
                               )
            self.model.compile(
                optimizer = Adam(lr = self.lr),
                # Note how we are passing the advantage and action layers to the
                # loss function
                loss = get_loss(
                    advantages = advantages, actions = actions, var = self.var,
                    loss_clipping_epsilon = self.loss_clipping_epsilon
                    )
                )

        # Define the callbacks for training. The callbacks write logs for the
        # training steps.

        self.callbacks = []
        # Add the callback for saving model data after regular intervals
        if self.model_saving_path is not None:
            callback = ModelCheckpoint(
                self.model_saving_path,
                period = self.model_saving_interval
                )
            self.callbacks.append(callback)

        # Add the callback for saving logs during training
        if self.training_logs_file_path is not None:
            callback = CSVLogger(self.training_logs_file_path, append = True)
            self.callbacks.append(callback)

    def get_policies(self, observations):
        """Return the means and variances for the multivariate Gaussian policy

        Arguments:
        observations -- A batch of observation. A numpy array of shape
                        (batch_size, n), where each observation is of shape
                        (n,)
        """

        # We will pass dummy data for advantages and actions during prediction
        # because the prediction does not depend on them
        model_input = [
                    observations,
                    np.array([[0.,] for i in range(observations.shape[0])]),
                    np.array([
                        [
                            0. for i in range(
                                self.env.action_space.shape[0]
                                )
                            ] for i in range(observations.shape[0])
                        ])
                    ]

        predictions = self.model.predict(model_input)
        vars = np.array([
            self.var for i in range(self.env.action_space.shape[0])
            ])
        policies = [
            {"means" : predictions[i], "vars" : vars}
            for i in range(predictions.shape[0])
            ]

        return policies

    def get_actions(self, policies):
        """Get actions given policies (means and variances)

        Arguments:
        policies -- A batch of policies. A numpy array of shape
                    (batch_size,), where each policy is a dictionary
                    {"means" : means, "vars" : vars}
        """

        # Sample from a Gaussian distribution with the given mean and var
        actions = np.array([
                [
                    norm(
                        loc = policies[i]["means"][j],
                        scale = sqrt(policies[i]["vars"][j])
                        ).rvs()
                    for j in range(self.env.action_space.shape[0])
                    ]
                for i in range(policies.shape[0])
            ])

        return actions

    def update_weights(self, observations, targets, advantages, actions,
                      minibatch_size, epochs
                      ):
        """Update weights of the model by learning from the data and return
        the loss

        Arguments:
        observations -- A batch of observation. A numpy array of shape
                        (batch_size, n), where each observation is of shape
                        (n,)
        targets -- The old predictions. We need this to compute surrogate loss
        advantages -- Finite horizon advantages
        action -- The actions chosen by the agent for this batch of observations
        minibatch_size -- Minibatch size
        epochs -- Number of epochs to train on the given data
        """

        loss = self.model.fit(
            [observations, advantages, actions],
            targets,
            batch_size = minibatch_size,
            epochs = epochs,
            callbacks = self.callbacks,
            verbose = False,
            )

        return loss


class DefaultCritic(object):
    """The default critic class for this reference implementation.

    The reference implementation assumes an actor critic architecture
    where the actor and the critic do not share parameters. It is also
    possible to have an architecture where the actor and the critic shares
    parameters. The loss defined in the original PPO paper (2017) assumes
    such an architecture. But this is out of the scope of this implementation
    as the PPO agent defined in ppo_agent.py assumes that the actor and the
    critic are distinct. We did this to make the implementation simpler.

    You can always write your own implementation of the critic model that
    differs from the default implementation. Just override the methods
    of this class, while keeping the methods signatures same. Then import your
    new class and use it in run_ppo_agent.py instead of this default critic.
    """

    def __init__(self, env, lr, model = None, training_logs_file_path = None,
                 model_saving_path = None, model_saving_interval = 1
                 ):
        """
        Arguments:
        env -- A Gym environment (can be vanilla or wrapped). We need this
               for infering input and output dimensions of the model.
        lr -- Learning rate
        model -- A compiled Keras model. If None, then the model is created.
        training_logs_file_path -- The file where we should save the logs for
                               training. This is useful for monitoring loss etc.
        model_saving_path -- The model will be saved to this filepath after a
                             certain number of epochs of training
        model_saving_interval -- The model will be saved to model_saving_path after
                                 this many epochs
        """
        self.env = env
        self.lr = lr
        self.training_logs_file_path = training_logs_file_path
        self.model_saving_path = model_saving_path
        self.model_saving_interval = model_saving_interval

        # Define the model
        if model is not None:
            self.model = model
        else:
            # If no model is specified, create one
            self.model = Sequential()
            self.model.add(
                Dense(
                    64,
                    input_dim = self.env.observation_space.shape[0],
                    activation = "relu"
                    )
                )
            self.model.add(Dense(64, activation = "relu"))
            self.model.add(Dense(1))

            self.model.compile(optimizer = Adam(lr = self.lr), loss = "mse")

        # Define the callbacks for training. The callbacks write logs for the
        # training steps.
        self.callbacks = []

        # Add the callback for saving model data after regular intervals
        if self.model_saving_path is not None:
            callback = ModelCheckpoint(
                self.model_saving_path,
                period = self.model_saving_interval
                )
            self.callbacks.append(callback)

        # Add the callback for saving logs during training
        if self.training_logs_file_path is not None:
            callback = CSVLogger(self.training_logs_file_path, append = True)
            self.callbacks.append(callback)

    def get_value(self, observations):
        """Return the value functions predicted by the critic for the given
        observations

        Arguments:
        observations -- A batch of observation. A numpy array of shape
                        (batch_size, n), where each observation is of shape
                        (n,)
        """

        values = self.model.predict(observations)

        return values

    def update_weights(self, observations, targets, minibatch_size, epochs):
        """Update weights of the model by learning from the data and return
        the loss

        Arguments:
        observations -- A batch of observation. A numpy array of shape
                        (batch_size, n), where each observation is of shape
                        (n,)
        targets -- Value function targets corresponding the observations. In
                   the reference implementation, the targets are computed
                   using a finite horizon version of TD(lambda)
        minibatch_size -- Minibatch size
        epochs -- Number of epochs to train on the given data
        """

        loss = self.model.fit(
            observations,
            targets,
            batch_size = minibatch_size,
            epochs = epochs,
            callbacks = self.callbacks,
            verbose = False,
            )

        return loss
