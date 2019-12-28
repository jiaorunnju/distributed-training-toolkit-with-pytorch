from abc import abstractmethod


class TrainTask:

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def get_model(self):
        """
        build a pytorch model
        :return: the model built
        """
        pass

    @abstractmethod
    def get_train_dataset(self, path):
        """
        get the train set
        :return: train set
        """
        pass

    @abstractmethod
    def get_valid_dataset(self, path):
        """
        get the valid set
        :return: valid set
        """
        pass

    @abstractmethod
    def get_test_dataset(self, path):
        """
        get the test set
        :return: test set
        """
        pass

    @abstractmethod
    def get_criterion(self):
        """
        make a function that compute the loss
        the loss function takes exactly 2 inputs, source and target
        :return: loss function
        """
        pass

    @abstractmethod
    def get_metric(self):
        """
        make a function that compute the metric,
        return a dict, name is the name of all metrics, and the function metric
        takes in source and target, returns values of all metrics in a list.
        :return: metric function
        """
        pass

    @abstractmethod
    def update_metric(self, old, new):
        """
        compare 2 metrics so we can keep the best model
        :param old: old metric value
        :param new: new metric value
        :return:  is_better: whether the new one is better; best: the best metric value
        """
        pass
