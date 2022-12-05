from PIL import Image
from sklearn.metrics import classification_report
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from variable_width_resnet import BasicBlock, VariableWidthResNet, AttentionResNet, AttentionedResNet
import json
import os
import pytorch_lightning as pl
import torch
import tqdm


torch.set_printoptions(threshold=10_000)

transform = transforms.Compose([transforms.Resize(120), transforms.ToTensor()])


class Main_Classifier(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = VariableWidthResNet(block=BasicBlock, layers=[2, 2, 2, 2], input_channels=3, width=64, num_classes=10)
    self.loss = nn.CrossEntropyLoss()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
    return {
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }


class Auxiliary_Classifier(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = AttentionResNet(block=BasicBlock, layers=[2, 2, 2, 2], input_channels=3, width=64, num_classes=3)
    self.loss = nn.CrossEntropyLoss()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
    return {
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }


class Attention_Classifier(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = AttentionedResNet(block=BasicBlock, layers=[2, 2, 2, 2], input_channels=3, width=64, num_classes=10)
    self.loss = nn.CrossEntropyLoss()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
    return {
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }


class BiasedMNIST(Dataset):
    "Biased MNIST dataset"
    def __init__(self, train: bool, corr: str="0.75", transform=None):
        """
        Args:
            train: Whether the loaded segment is for training.
            corr: The correlation ratio of biases and labels. 0 is the least and 1 is the greatest. Provided values are 0.1, 0.5, 0.75, 0.9, 0.95, and 0.99.
            transform: Transforms to be applied on a sample.
        """
        self.root_dir = "/gpfsnyu/scratch/rm5327"
        self.transform = transform
        if train:
            self.img_path = os.path.join(self.root_dir, "biased_mnist", "full_"+corr, "trainval")
            self.ann_path = os.path.join(self.root_dir, "biased_mnist", "full_"+corr, "trainval.json")
            with open(self.ann_path) as annotations:
                self.ann = json.load(annotations)
        else:
            self.img_path = os.path.join(self.root_dir, "biased_mnist", "full", "test")
            self.ann_path = os.path.join(self.root_dir, "biased_mnist", "full", "test.json")
            with open(self.ann_path) as annotations:
                self.ann = json.load(annotations)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.img_path, str(idx)+".jpg"))
        label = self.ann[idx]["digit"]

        if self.transform:
            image = self.transform(image)

        return image, label


class Two_class(Dataset):
    def __init__(self, corr, mean, std, model, transform):
        """
        Args:
            train: Whether the loaded segment is for training.
            corr: The correlation ratio of biases and labels. 0 is the least and 1 is the greatest. Provided values are 0.1, 0.5, 0.75, 0.9, 0.95, and 0.99.
            transform: Transforms to be applied on a sample.
        """
        self.root_dir = "/gpfsnyu/scratch/rm5327"
        self.transform = transform
        self.img_path = os.path.join(self.root_dir, "biased_mnist", "full_"+corr, "trainval")
        self.ann_path = os.path.join(self.root_dir, "biased_mnist", "full_"+corr, "trainval.json")
        with open(self.ann_path) as annotations:
                self.ann = json.load(annotations)
        self.model = model
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.img_path, str(idx)+".jpg"))

        if self.transform:
            image = self.transform(image)

        logits, _ = get_prediction(image.unsqueeze(0), self.model)
        standardized = (logits[0] - self.mean) / self.std

        return image, (standardized > 0.5) * 2 + (standardized < -0.5) * 1


def get_prediction(x, model: pl.LightningModule):
	model.freeze()
	probabilities = torch.softmax(model(x), dim=1)
	predicted_class = torch.argmax(probabilities, dim=1)
	return predicted_class, probabilities


def first_step(corr):
    train_biasedMNIST = BiasedMNIST(train=True, corr=f"{corr}", transform=transform)
    train_bM = DataLoader(train_biasedMNIST, batch_size=128, shuffle=True, num_workers=12)
    test_biasedMNIST = BiasedMNIST(train=False, transform=transform)
    test_bM = DataLoader(test_biasedMNIST, batch_size=128, shuffle=False, num_workers=12)
    model = Main_Classifier()
    trainer = pl.Trainer(accelerator='gpu', devices=-1, max_epochs=90)
    trainer.fit(model, train_bM)

    trainer.save_checkpoint(f"modified_resnet18_{corr}_2.pt")

    inference_model = Main_Classifier().load_from_checkpoint(f"modified_resnet18_{corr}_2.pt", map_location=torch.device('cuda'))

    true_y, pred_y = [], []

    for batch in tqdm.tqdm(iter(test_bM), total=len(test_bM)):
        x, y = batch
        true_y.extend(y)
        preds, probs = get_prediction(x, inference_model)
        pred_y.extend(preds.cpu())

    print(classification_report(true_y, pred_y, digits=3))


def second_step(corr):

    train_biasedMNIST = BiasedMNIST(train=True, corr=str(corr), transform=transform)
    train_bM = DataLoader(train_biasedMNIST, batch_size=30000, shuffle=True, num_workers=12)

    main_model = Main_Classifier().load_from_checkpoint(f"modified_resnet18_0.9.pt", map_location=torch.device('cuda'))
    main_model.freeze()

    samples, label = next(iter(train_bM))
    post_conv = torch.max(main_model(samples), dim=1)[0]
    zeros = torch.mul(post_conv, label == 0)[torch.mul(post_conv, label == 0).nonzero()]
    mean, std = torch.mean(zeros), torch.std(zeros)

    tc = Two_class(corr=str(corr), mean=mean, std=std, model=main_model, transform=transform)
    two_class = DataLoader(tc, batch_size=128, shuffle=True, num_workers=12)

    main_model = Main_Classifier().load_from_checkpoint(f"modified_resnet18_0.9.pt")
    auxiliary_model = Auxiliary_Classifier()

    auxiliary_model.model.conv1 = main_model.model.conv1
    auxiliary_model.model.bn1 = main_model.model.bn1
    auxiliary_model.model.layer1 = main_model.model.layer1
    auxiliary_model.model.layer2 = main_model.model.layer2
    auxiliary_model.model.layer3 = main_model.model.layer3
    auxiliary_model.model.layer4 = main_model.model.layer4
    auxiliary_model.freeze()
    auxiliary_model.model.mlp = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 512), nn.Sigmoid())
    auxiliary_model.model.fc = nn.Linear(8 * 64 * 1, 3)

    trainer = pl.Trainer(accelerator='gpu', devices=-1, max_epochs=20)
    trainer.fit(auxiliary_model, two_class)

    trainer.save_checkpoint(f"auxiliary_{corr}.pt")


def third_step(corr):
    train_biasedMNIST = BiasedMNIST(train=True, corr=f"{corr}", transform=transform)
    train_bM = DataLoader(train_biasedMNIST, batch_size=128, shuffle=True, num_workers=12)
    test_biasedMNIST = BiasedMNIST(train=False, transform=transform)
    test_bM = DataLoader(test_biasedMNIST, batch_size=128, shuffle=False, num_workers=12)

    vanila_model = Main_Classifier().load_from_checkpoint(f"modified_resnet18_{corr}.pt")
    auxiliary_model = Auxiliary_Classifier().load_from_checkpoint(f"auxiliary_{corr}.pt")
    main_model = Attention_Classifier()

    main_model.model.conv1 = vanila_model.model.conv1
    main_model.model.bn1 = vanila_model.model.bn1
    main_model.model.layer1 = vanila_model.model.layer1
    main_model.model.layer2 = vanila_model.model.layer2
    main_model.model.layer3 = vanila_model.model.layer3
    main_model.model.layer4 = vanila_model.model.layer4

    main_model.model.auxiliary_mlp = auxiliary_model.model.mlp
    main_model.model.conv1.req_grad = False
    main_model.model.bn1.req_grad = False
    main_model.model.layer1[0].conv1.req_grad = False
    main_model.model.layer1[0].bn1.req_grad = False
    main_model.model.layer1[0].conv2.req_grad = False
    main_model.model.layer1[0].bn2.req_grad = False
    main_model.model.layer1[1].conv1.req_grad = False
    main_model.model.layer1[1].bn1.req_grad = False
    main_model.model.layer1[1].conv2.req_grad = False
    main_model.model.layer1[1].bn2.req_grad = False
    main_model.model.layer2[0].conv1.req_grad = False
    main_model.model.layer2[0].bn1.req_grad = False
    main_model.model.layer2[0].conv2.req_grad = False
    main_model.model.layer2[0].bn2.req_grad = False
    main_model.model.layer2[0].downsample[0].req_grad = False
    main_model.model.layer2[0].downsample[1].req_grad = False
    main_model.model.layer2[1].conv1.req_grad = False
    main_model.model.layer2[1].bn1.req_grad = False
    main_model.model.layer2[1].conv2.req_grad = False
    main_model.model.layer2[1].bn2.req_grad = False
    main_model.model.layer3[0].conv1.req_grad = False
    main_model.model.layer3[0].bn1.req_grad = False
    main_model.model.layer3[0].conv2.req_grad = False
    main_model.model.layer3[0].bn2.req_grad = False
    main_model.model.layer3[0].downsample[0].req_grad = False
    main_model.model.layer3[0].downsample[1].req_grad = False
    main_model.model.layer3[1].conv1.req_grad = False
    main_model.model.layer3[1].bn1.req_grad = False
    main_model.model.layer3[1].conv2.req_grad = False
    main_model.model.layer3[1].bn2.req_grad = False
    main_model.model.layer4[0].conv1.req_grad = False
    main_model.model.layer4[0].bn1.req_grad = False
    main_model.model.layer4[0].conv2.req_grad = False
    main_model.model.layer4[0].bn2.req_grad = False
    main_model.model.layer4[0].downsample[0].req_grad = False
    main_model.model.layer4[0].downsample[1].req_grad = False
    main_model.model.layer4[1].conv1.req_grad = False
    main_model.model.layer4[1].bn1.req_grad = False
    main_model.model.layer4[1].conv2.req_grad = False
    main_model.model.layer4[1].bn2.req_grad = False
    main_model.model.auxiliary_mlp[0].req_grad = False
    main_model.model.auxiliary_mlp[2].req_grad = False
    main_model.model.fc = nn.Linear(8 * 64 * 1, 10)

    trainer = pl.Trainer(accelerator='gpu', devices=-1, max_epochs=90)
    trainer.fit(main_model, train_bM)

    true_y, pred_y = [], []

    for batch in tqdm.tqdm(iter(test_bM), total=len(test_bM)):
        x, y = batch
        true_y.extend(y)
        preds, probs = get_prediction(x, main_model)
        pred_y.extend(preds.cpu())

    print(classification_report(true_y, pred_y, digits=3))



if __name__ == "__main__":
    first_step(0.9)
    # second_step(0.9)
    # third_step(0.9)
