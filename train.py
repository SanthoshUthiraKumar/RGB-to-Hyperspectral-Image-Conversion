import os
import numpy as np
import scipy.io as sio
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models import ResNetGenerator, NLayerDiscriminator

# --- CONFIG ---
DATA_PATH = "./dataset"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

class AgroDataset(Dataset):
    def __init__(self, data_folder):
        self.files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
        self.data_folder = data_folder

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_folder, self.files[idx])
        hs_data = None
        try:
            # Try Standard
            mat = sio.loadmat(path)
            key = [k for k in mat.keys() if not k.startswith('__')][0]
            hs_data = mat[key]
        except:
            # Try H5PY (New Format)
            try:
                with h5py.File(path, 'r') as f:
                    key = list(f.keys())[0]
                    hs_data = np.transpose(np.array(f[key]), (2, 1, 0))
            except: return self.__getitem__((idx + 1) % len(self))

        # Fix Shape (H, W, C)
        if hs_data.shape[0] < hs_data.shape[2]: hs_data = np.transpose(hs_data, (1, 2, 0))
        
        # Extract RGB (Approx bands 29, 15, 5)
        bands = hs_data.shape[2]
        rgb = hs_data[:, :, [min(29, bands-1), min(15, bands-1), min(5, bands-1)]]

        # Normalize to [-1, 1]
        max_val = 4095.0 if hs_data.max() > 255 else 1.0
        hs_data = (hs_data.astype(np.float32) / max_val) * 2.0 - 1.0
        rgb = (rgb.astype(np.float32) / max_val) * 2.0 - 1.0

        # To Tensor
        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1).float(),
            "hs": torch.from_numpy(hs_data).permute(2, 0, 1).float()
        }

if __name__ == "__main__":
    # Check bands in dataset
    temp = AgroDataset(DATA_PATH)
    bands = temp[0]['hs'].shape[0]
    print(f"Training for {bands} bands...")

    netG = ResNetGenerator(output_nc=bands).to(DEVICE)
    netD = NLayerDiscriminator(input_nc=3+bands).to(DEVICE)
    opt_G = optim.Adam(netG.parameters(), lr=0.0002)
    opt_D = optim.Adam(netD.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    dataloader = DataLoader(temp, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            real_rgb = batch['rgb'].to(DEVICE)
            real_hs = batch['hs'].to(DEVICE)

            # Train G
            opt_G.zero_grad()
            fake_hs = netG(real_rgb)
            loss_G = criterion(netD(torch.cat((real_rgb, fake_hs), 1)), torch.ones(real_rgb.size(0), 1, 62, 62).to(DEVICE)) + \
                     criterion(fake_hs, real_hs) * 100
            loss_G.backward()
            opt_G.step()

            # Train D
            opt_D.zero_grad()
            loss_D = (criterion(netD(torch.cat((real_rgb, real_hs), 1)), torch.ones(real_rgb.size(0), 1, 62, 62).to(DEVICE)) + \
                      criterion(netD(torch.cat((real_rgb, fake_hs.detach()), 1)), torch.zeros(real_rgb.size(0), 1, 62, 62).to(DEVICE))) * 0.5
            loss_D.backward()
            opt_D.step()

        print(f"Epoch {epoch} Complete. Loss G: {loss_G.item():.4f}")
        torch.save(netG.state_dict(), f"{CHECKPOINT_DIR}/netG_final.pth")


