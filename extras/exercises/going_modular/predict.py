import torch
from torchvision import transforms
from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import random
from pathlib import Path
import argparse
import model, dataset
from omegaconf import OmegaConf
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='D:/VScodeProjects/pytorch-deep-learning/extras/exercises/going_modular/config.yaml',
                    type=str,
                    help="config")
# Add a new argument for the image path
parser.add_argument("--image",
                    type=str,
                    help="Path to the image for prediction")

data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

# 1. Take in a trained model, class names, image path, image size, a transform and a target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        transform: transforms=None):
    
    # 2. Open image
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The image file {image_path} was not found.")
        return
    
    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = data_transform

    ### Predict on image ###

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # 6. Transform and add an extra dimension to imgae 
        transformed_image = image_transform(img).unsqueeze(dim=0)
        # 7. Make a prediction on image with an extra dimension and send to target device
        pred_logits = model(transformed_image.to(device))
        # 8. logits -> probabilities
        pred_prob = torch.softmax(pred_logits, dim=1)
        # 9. probabilities -> prediction labels
        pred_label = torch.argmax(pred_prob, dim=1)

        # 10. Plot image with label and probabiliy
        plt.figure()
        plt.imshow(img)
        plt.title(f"Pred: {class_names[pred_label.item()]} | Prob: {pred_prob.max():.3f}")
        plt.axis(False)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    work_dir = cfg.work_dir.dir
    train_dir = cfg.dataset.train_dir
    test_dir = cfg.dataset.test_dir

    batch_size = cfg.train.batch_size

    train_dataloader, test_dataloader, class_names = dataset.create_dataloader(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                transform=data_transform,
                                                                                batch_size= batch_size,
                                                                                num_workers=os.cpu_count())
            

    model_1 = model.TinyVGG(input_shape=3,
                            hidden_units=128,
                            output_shape=len(class_names))
    
    state_dict_path = "D:/VScodeProjects/pytorch-deep-learning/extras/exercises/going_modular/work_dir/checkpoints/2025-04-30-19-09/TinyVGG_ex5.pth"
    # Set weights_only=True to avoid the future warning
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model_1.load_state_dict(state_dict=state_dict)

    if args.image:
        pred_and_plot_image(model=model_1, 
                            image_path=args.image,
                            class_names=class_names,
                            transform=data_transform)
        plt.show()
    else:
        model_1.to(device)
        # Get a random list of image paths from test set
        num_images_to_plot = 1
        test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
        test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                            k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

        # Make predictions on and plot the images
        for image_path in test_image_path_sample:
            pred_and_plot_image(model=model_1, 
                                image_path=str(image_path),
                                class_names=class_names,
                                transform=data_transform)
        plt.show()