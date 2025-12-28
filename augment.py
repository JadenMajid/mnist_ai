import os
import struct
import numpy as np
import torch
from torchvision import transforms

def read_idx(path):
    with open(path, 'rb') as f:
        # Read the magic number to determine if it's images or labels
        magic, num = struct.unpack(">II", f.read(8))
        if magic == 2051:  # Image file
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
            return data, (magic, num, rows, cols)
        else:  # Label file
            data = np.fromfile(f, dtype=np.uint8)
            return data, (magic, num)

def write_idx(path, data, header):
    with open(path, 'wb') as f:
        if header[0] == 2051:  # Images
            # Header: Magic, Count, Rows, Cols
            f.write(struct.pack(">IIII", *header))
        else:  # Labels
            # Header: Magic, Count
            f.write(struct.pack(">II", *header))
        f.write(data.astype(np.uint8).tobytes())

def augment_set(images, labels):
    """Applies random transforms to the input set."""
    augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(12),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])
    
    aug_images = np.zeros_like(images)
    for i in range(len(images)):
        # Transform and convert back to 0-255 uint8 range
        img_tensor = augmenter(images[i])
        aug_images[i] = (img_tensor.numpy().squeeze() * 255).astype(np.uint8)
    
    return aug_images

def process_mnist(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    file_pairs = [
        ('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'),
        ('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    ]

    for img_file, lbl_file in file_pairs:
        print(f"Processing {img_file}...")
        
        # 1. Load
        images, img_header = read_idx(os.path.join(input_dir, img_file))
        labels, lbl_header = read_idx(os.path.join(input_dir, lbl_file))
        
        # 2. Augment
        # Note: We keep the labels the same as the images are just transformed
        aug_images = augment_set(images, labels)
        
        # 3. Save with identical headers
        write_idx(os.path.join(output_dir, img_file), aug_images, img_header)
        write_idx(os.path.join(output_dir, lbl_file), labels, lbl_header)

if __name__ == "__main__":
    # Your current directory structure
    SRC = "./archive"
    DEST = "./augmented_mnist"
    
    process_mnist(SRC, DEST)
    print(f"\nDone! Augmented files saved to: {DEST}")