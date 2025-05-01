from typing import Dict, List, Tuple
import torch
import torch.utils
import torch.utils.data
from torchvision import transforms
from tqdm.auto import tqdm
import argparse
from omegaconf import OmegaConf
import datetime
import os

import model
import dataset


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='D:/VScodeProjects/pytorch-deep-learning/extras/exercises/going_modular/config.yaml',
                    type=str,
                    help="config")
parser.add_argument("--num_epochs", default=5, type=int, help="num epochs")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--hidden_units", default=128, type=int, help="hidden units")
parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch
    """
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through dataloader 
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    # Put a model in eval mode
    model.eval()

    # Setup test loss and accuracy
    test_loss, test_acc = 0, 0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop through dataloader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)
            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            # 3. Calculate and accumulate accuracy
            test_pred_labels = y_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    # Creat empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        print(
            f"Epoch: {epoch} |"
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} |"
            f"test_loss: {test_loss:.4f} |"
            f"test_acc: {test_acc:.4f} |"
        )

        #Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    work_dir = cfg.work_dir.dir
    train_dir = cfg.dataset.train_dir
    test_dir = cfg.dataset.test_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.checkpoint_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.checkpoint_dir, timestamp)

    os.makedirs(cfg.work_dir.checkpoint_dir, exist_ok=True)

    train_dataloader, test_dataloader, class_names = dataset.create_dataloader(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=data_transform,
                                                                               batch_size= args.batch_size,
                                                                               num_workers=os.cpu_count())
    model_1 = model.TinyVGG(input_shape=3,
                            hidden_units=args.hidden_units,
                            output_shape=len(class_names))
    model_1 = model_1.to(device)
    
    optimizer = torch.optim.Adam(model_1.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = cfg.train.num_epochs

    # setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    train(model=model_1,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          optimizer=optimizer,
          loss_fn=loss_fn,
          epochs=epochs,
          device=device)
    
    model_name = "TinyVGG_ex5.pth"
    model_save_path = os.path.join(cfg.work_dir.checkpoint_dir, model_name)
    torch.save(obj=model_1.state_dict(),
               f=model_save_path)