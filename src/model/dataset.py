import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy black image or handle gracefully
            image = Image.new('RGB', (224, 224))
            
        # Label: 1 for dog, 0 for cat
        # Assuming filename format like "cat.0.jpg" or "dog.123.jpg"
        label = 1.0 if 'dog' in img_name.lower() else 0.0

        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            default_tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image = default_tf(image)

        return image, label
