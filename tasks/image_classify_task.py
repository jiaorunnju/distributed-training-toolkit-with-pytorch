import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .train_task import TrainTask


class ImageClassifyTask(TrainTask):

    def __init__(self, cfg):
        super().__init__(cfg)

    def get_model(self):
        return models.mobilenet_v2()

    def get_train_dataset(self, path):
        train_dir = self.cfg.TRAIN.TRAIN_DATA
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        return train_dataset

    def get_valid_dataset(self, path):
        valid_dir = self.cfg.TRAIN.VALID_DATA
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        valid_dataset = datasets.ImageFolder(
            valid_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        )
        return valid_dataset

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
            "name": ['Acc@1', 'Acc@5'],
            "metric": lambda x, y: accuracy(x, y, topk=(1, 5))
        }
