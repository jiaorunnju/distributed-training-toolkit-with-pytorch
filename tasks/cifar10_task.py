import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .train_task import TrainTask


class Cifar10Task(TrainTask):

    def get_model(self):
        return models.resnet18()

    def get_train_dataset(self, path):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return datasets.CIFAR10(path, train=True, transform=transform_train)

    def get_valid_dataset(self, path):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return datasets.CIFAR10(path, train=False, transform=transform_test)

    def get_test_dataset(self, path):
        pass

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def get_metric(self):
        def accuracy(output, target, topk=(1,)):
            """Computes the accuracy over the k top predictions for the specified values of k"""
            with torch.no_grad():
                maxk = max(topk)
                batch_size = target.size(0)

                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))

                res = []
                for k in topk:
                    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(100.0 / batch_size))
                return res

        return {
            "name": ['Acc@1', ],
            "metric": lambda x, y: accuracy(x, y, topk=(1,)),

        }

    def update_metric(self, old, new):
        return new > old, max(old, new)
