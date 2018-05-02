class QValueNotFoundError(Exception):
    """An custom error that indicates that the table has no information about
    the Q value for a given observation and action. Essentially, a KeyError for
    the Q value table.
    """
    pass


class DefaultTable(object):
    """A table that stores Q values for state action pairs

    You can always write your own implementation that
    differs from the default implementation. Just override the methods
    of this class, while keeping the methods signatures same. Then import your
    new class and use it in run_glie_monte_carlo_agent.py instead of this
    default q value table.
    """

    def __init__(self, q_value_table = None, q_value_table_file_path = None):
        """
        Arguments:
        q_value_table -- A dictionary with state action pair as key and
                         Q value as the value. Pass a saved dict if you want
                         to test the performance of a trained agent. Otherwise,
                         you don't need to pass this argument.
        q_value_table_file_path -- The file path where the save() method will
                                   save the information contained in thr Q value
                                   dict. You don't need to pass this argument if
                                   you never intend to call the save() method.
        """

        if q_value_table is not None:
            self.q_value_table = q_value_table
        else:
            # If saved dict wasn't provided, we start from scratch.
            self.q_value_table = {}

        self.q_value_table_file_path = q_value_table_file_path

    def get_key_from_observation_and_action(self, observation, action):
        """Get the key of the Q value dict from the observation and action

        Arguments:
        observation -- Gym observation. Type may vary.
        action -- Gym action. Type may vary.

        Notes:
        The observation and action in the Gym environment may not be hashable.
        But a dictionary key needs to be hashable. So we put the observation
        and action together into a hashable object.
        """

        key = (tuple(observation), action)

        return key

    def get_q_value(self, observation, action):
        """Get the Q value for the observation and action from the Q value dict

        Arguments:
        observation -- Gym observation
        action -- Gym action

        Raises:
        QValueNotFoundError -- if there's no information about this observation
                               action pair in the Q value dict.
        """

        key = self.get_key_from_observation_and_action(observation, action)

        try:
            q_value = self.q_value_table[key]
            return q_value
        except KeyError:
            message = (
                "Q value table doesn't have any data on the combination "
                "Observation : {0} and Action : {1}".format(observation, action)
                )
            raise QValueNotFoundError(message)

    def update_q_value(self, observation, action, q_value):
        """Update the Q value for the observation action pair in the Q value
        dict

        Arguments:
        observation -- Gym observation
        action -- Gym action
        q_value -- Update to this value
        """

        key = self.get_key_from_observation_and_action(observation, action)

        self.q_value_table[key] = q_value

    def save(self):
        """Save the information in the Q value dict to file"""

        # If the class wasn't instantiated with file paths, complain.
        if self.q_value_table_file_path is None:
            raise Exception(
                "File path for saving Q value table was not specified."
                )

        # It would be easier for the viewer of the saved file to understand
        # the data if the keys were sorted.
        sorted_state_action_pairs = sorted([key for key in self.q_value_table])

        with open(self.q_value_table_file_path, "w") as q_value_table_fh:
            for state_action_pair in sorted_state_action_pairs:
                q_value_table_fh.write(
                    "{0}\t{1}\n".format(
                        state_action_pair, self.q_value_table[state_action_pair]
                        )
                    )
