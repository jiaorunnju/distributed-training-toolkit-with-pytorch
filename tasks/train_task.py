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
        :return: loss function
        """
        pass

    @abstractmethod
    def get_metric(self):
        """
        make a function that compute the metric
        :return: metric function
        """
        pass
