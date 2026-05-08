import os
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from DiffusionFreeGuidence.ModelCondition import UNet  
from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, DDIMSampler
import torch.nn.functional as F  
torch.manual_seed(1234)

#############################################
# 1. Utility function to compute profiles
#############################################
def compute_energy_profiles(images):
    """
    Given a batch of images (assumed to be 2-channel: xz view and yz view),
    compute the energy deposit profiles along the x-axis (from xz view) and 
    the z-axis (averaged from both views).

    Assumes images are normalized in [-1, 1] and have shape (B, 2, H, W).
    Returns:
      profile_x: 1D array of length W
      profile_y: 1D array of length W (unused here)
      profile_z: 1D array of length H
    """
    images = (images + 1) / 2.0  # Convert to [0,1]
    batch, channels, H, W = images.shape

    if channels == 2:
        profile_x = images[:, 0, :, :].sum(dim=2)  # shape (B, W)
        profile_y = images[:, 1, :, :].sum(dim=2)  # shape (B, W)
        profile_z_xz = images[:, 0, :, :].sum(dim=1)  # shape (B, H)
        profile_z_yz = images[:, 1, :, :].sum(dim=1)  # shape (B, H)
        profile_z = 0.5 * (profile_z_xz + profile_z_yz)
    else:
        raise ValueError("Expected images with 2 channels.")
    
    return profile_x.mean(dim=0).cpu().numpy(), profile_y.mean(dim=0).cpu().numpy(), profile_z.mean(dim=0).cpu().numpy()

#############################################
# 2. Function to compute real data profiles by labels (including material)
#############################################
def get_real_energy_profiles_by_labels(h5_file, energy_val, xy_val, z_val, material_value, img_size, device, batch_size=1000, tol=0.05):
    """
    Opens the HDF5 file and filters images whose stored energy, xy, z, and material 
    conditions match the given ones (within a tolerance for xy and z).
    Material is filtered by converting the stored bytes to a string, stripping whitespace,
    and then mapping using:
         {"PbF2": 0.0, "PbWO4": 1.0}
    """
    # Define material mapping here (should match the one used during training)
    material_mapping = {"PbF2": 0.0, "PbWO4": 1.0}
    
    with h5py.File(h5_file, 'r') as f:
        images_xz = f['images_xz'][:]   # shape: (N, H, W)
        images_yz = f['images_yz'][:]   # shape: (N, H, W)
        energy_labels = f['labels_energy'][:]
        xy_labels_raw = f['labels_xy'][:]   # e.g. 1,2,3,4,5
        z_labels_raw = f['labels_z'][:]       # e.g. 4,6,8,10,15
        material_labels_raw = f['labels_material'][:]
        # Decode and map material labels
        material_labels = np.array([
            material_mapping[(m.decode('utf-8').strip() if isinstance(m, bytes) else m.strip())]
            for m in material_labels_raw
        ], dtype=np.float32)
        print("Unique energy labels:", np.unique(energy_labels))
    
    xy_labels = (xy_labels_raw - 1) / (5 - 1)    # normalized to [0,1]
    z_labels  = (z_labels_raw - 4) / (15 - 4)      # normalized to [0,1]

    
    # Select indices satisfying energy, xy, z, and material conditions.
    idx = np.where((energy_labels == energy_val) &
                   (np.abs(xy_labels - xy_val) < tol) &
                   (np.abs(z_labels - z_val) < tol) &
                   (material_labels == material_value))[0]
    if len(idx) == 0:
        raise ValueError(f"No images found for energy {energy_val}, xy {xy_val}, z {z_val}, material {material_value}")
    idx = idx[:batch_size]

    imgs_xz = images_xz[idx]
    imgs_yz = images_yz[idx]
    imgs_xz = torch.tensor(imgs_xz, dtype=torch.float32).unsqueeze(1)  # (B, 1, H, W)
    imgs_yz = torch.tensor(imgs_yz, dtype=torch.float32).unsqueeze(1)
    # Convert from [0,1] to [-1,1]
    imgs_xz = imgs_xz / 0.5 - 1.0
    imgs_yz = imgs_yz / 0.5 - 1.0
    imgs_xz = F.interpolate(imgs_xz, size=(img_size, img_size), mode='bilinear', align_corners=False)
    imgs_yz = F.interpolate(imgs_yz, size=(img_size, img_size), mode='bilinear', align_corners=False)
    imgs = torch.cat([imgs_xz, imgs_yz], dim=1).to(device)
    return compute_energy_profiles(imgs)

#############################################
# 3. Function to load model and its sampler
#############################################
def load_model_and_sampler(checkpoint_path, modelConfig, device):
    from DiffusionFreeGuidence.ModelCondition import UNet  
    model = UNet(
        T=modelConfig["T"],
        num_energy_labels=11,
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation!")
        model = torch.nn.DataParallel(model)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    sampler = DDIMSampler(
        model,
        beta_1=modelConfig["beta_1"],
        beta_T=modelConfig["beta_T"],
        T=modelConfig["T"],
        eta=0.0,
        ddim_steps=50
    ).to(device)
    return model, sampler

#############################################
# 4. Function to generate profiles for given labels (including material)
#############################################
def get_generated_energy_profiles_for_labels(sampler, modelConfig, device, energy_label, xy_value, z_value, material_value=0.0, batch_size=1000):
    noise = torch.randn(batch_size, 1, modelConfig["img_size"], modelConfig["img_size"], device=device)
    energy_tensor = torch.full((batch_size,), energy_label, dtype=torch.long, device=device)
    xy_tensor = torch.full((batch_size,), xy_value, dtype=torch.float32, device=device)
    z_tensor = torch.full((batch_size,), z_value, dtype=torch.float32, device=device)
    material_tensor = torch.full((batch_size,), material_value, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        gen_imgs = sampler(noise, energy_tensor, xy_tensor, z_tensor, material_tensor)
    # Assume generated images are single channel; replicate to 2 channels
    gen_imgs = gen_imgs.repeat(1, 2, 1, 1)
    return compute_energy_profiles(gen_imgs)


def plot_profiles_for_cell_sizes(real_profiles_dict, gen_profiles_dict, energy_values, cell_sizes, material_mapping, save_dir=None):

    xy_mapping = {0.0: 1, 0.25: 2, 0.5: 3, 0.75: 4, 1.0: 5}
    z_mapping = {0.0: 4, 0.0909: 5, 0.3636: 8, 0.5455: 10, 1.0: 15}

    for material_val, material_str in material_mapping.items():
        for (xy, z) in cell_sizes:

            fig, axs = plt.subplots(
                4,
                len(energy_values),
                figsize=(4 * len(energy_values), 12),
                gridspec_kw={'height_ratios':[3,1,3,1]},
            )

            for j, energy in enumerate(energy_values):

                key = (energy, xy, z, material_val)

                if key not in real_profiles_dict or key not in gen_profiles_dict:
                    print(f"No data for key: {key}")
                    continue

                real_x, _, real_z = real_profiles_dict[key]
                gen_x, _, gen_z = gen_profiles_dict[key]

                # convert normalized cell size to physical size
                raw_xy = xy_mapping[xy]
                raw_z  = z_mapping[z]

                # calorimeter size (5 cells in each direction)
                tot_x = raw_xy * 5
                tot_z = raw_z * 5

                # physical coordinate axis
                x_axis = np.linspace(0, tot_x, len(real_x))
                z_axis = np.linspace(0, tot_z, len(real_z))

                # ---------- ratios ----------
                ratio_x = gen_x / (real_x + 1e-8)
                ratio_z = gen_z / (real_z + 1e-8)

                # compute symmetric limits around 1
                # limits for transverse ratio
                max_dev_x = np.max(np.abs(ratio_x - 1))
                ylim_low_x = 1 - (max_dev_x + 0.02)
                ylim_high_x = 1 + (max_dev_x + 0.02)

                # limits for longitudinal ratio
                max_dev_z = np.max(np.abs(ratio_z - 1))
                ylim_low_z = 1 - (max_dev_z + 0.02)
                ylim_high_z = 1 + (max_dev_z + 0.02)

                # ---------- Z PROFILE ----------
                axs[0, j].plot(z_axis, real_z, label='Ground truth', color='blue')
                axs[0, j].plot(z_axis, gen_z, label='Generated', color='red')
                axs[0, j].set_yscale('log') ###############################################
                axs[0, j].set_title(f"{energy} GeV", fontsize=16)
                axs[0, j].grid(True)

                # ---------- Z RATIO ----------
                axs[1, j].plot(z_axis, ratio_z, color='black')
                axs[1, j].axhline(1.0, linestyle='--')
                axs[1, j].set_ylim(ylim_low_z, ylim_high_z)
                axs[1, j].grid(True)

                # ---------- X PROFILE ----------
                axs[2, j].plot(x_axis, real_x, label='Ground truth', color='blue')
                axs[2, j].plot(x_axis, gen_x, label='Generated', color='red')
                axs[2, j].set_yscale('log') ###############################################
                axs[2, j].grid(True)

                # ---------- X RATIO ----------
                axs[3, j].plot(x_axis, ratio_x, color='black')
                axs[3, j].axhline(1.0, linestyle='--')
                axs[3, j].set_ylim(ylim_low_x, ylim_high_x)
                axs[3, j].grid(True)

                # labels only on first column
                if j == 0:
                    axs[0, j].set_ylabel("Energy deposition (a.u.)", fontsize=16)
                    axs[1, j].set_ylabel("Gen/Real", fontsize=16)
                    axs[2, j].set_ylabel("Energy deposition (a.u.)", fontsize=16)
                    axs[3, j].set_ylabel("Gen/Real", fontsize=16)

                axs[3, j].set_xlabel("Coordinate (cm)", fontsize=16)

                if j == 0:
                    axs[0, j].legend(fontsize=14)

            def find_nearest(mapping, value):
                key = min(mapping.keys(), key=lambda k: abs(k - value))
                return mapping[key]

            raw_xy = find_nearest(xy_mapping, xy)
            raw_z  = find_nearest(z_mapping,  z)

            fig.suptitle(
                f"Cell configuration {raw_xy}×{raw_xy}×{raw_z} cm³",
                fontsize=20
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                fig_path = os.path.join(save_dir, f"log_energy_profiles_{material_str}_xy{xy_mapping.get(xy,xy)}_z{z_mapping.get(z,z)}_small.png")
                plt.savefig(fig_path)
                print(f"Saved figure for {material_str} with xy = {xy_mapping.get(xy,xy)} cm, z = {z_mapping.get(z,z)} cm at {fig_path}")
            plt.show()


#############################################
# 6. Main evaluation and plotting
#############################################
if __name__ == "__main__":
    # Model configuration
    modelConfig = {
        "epoch": 1000,
        "batch_size": 1000,
        "T": 500,
        "channel": 32,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 3.,
        "device": "cuda:0",
        "save_weight_dir": "./CheckpointsCondition/", 
    }
    
    device = torch.device(modelConfig["device"])
    h5_file = "total_photon_shower.h5"
    checkpoint_path = os.path.join(modelConfig["save_weight_dir"], "ckpt_1800.pt") #good epoch 1800
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    #Define energy values (in GeV) and mappings
    # energy_values_raw = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # raw_energy_to_label = {
    #     1: 0,  10: 1, 20: 2, 30: 3, 40: 4,
    #     50: 5, 60: 6, 70: 7, 80: 8, 90: 9, 100: 10
    # }
    energy_values_raw = [1, 10, 50, 70, 100]

    raw_energy_to_label = {
    1: 0,  10: 1, 50: 5, 70: 7, 100: 10
}


    # # Cell sizes in normalized units (as used during training)
    cell_sizes = [
        (0.0, 0.0909),
        (0.25, 0.0),
        (0.5, 0.3636),
        (0.75, 0.5455),
        (1.0, 1.0)
    ]


    # Cell sizes in normalized units (as used during training)
    # cell_sizes = [
    #     (0.75, 0.5455),
    # ]
    
    # Define material conditions: 0.0 -> "PbF2", 1.0 -> "PbWO4"
    material_conditions = [0.0]#, 1.0]
    material_mapping = {0.0: "PbF2"}#, 1.0: "PbWO4"}
    
    # Compute real profiles and store in a dict with keys: (energy, xy, z, material)
    real_profiles_dict = {}
    for energy in energy_values_raw:
        mapped_energy = raw_energy_to_label[energy]
        for (xy_val, z_val) in cell_sizes:
            for material in material_conditions:
                key = (energy, xy_val, z_val, material)
                try:
                    profiles = get_real_energy_profiles_by_labels(
                        h5_file, mapped_energy, xy_val, z_val, material, modelConfig["img_size"], device)
                    real_profiles_dict[key] = profiles
                    print(f"Computed real profiles for energy {energy} GeV, xy {xy_val}, z {z_val}, material {material}.")
                except Exception as e:
                    print(f"Skipping key {key}: {e}")
                finally:
                    torch.cuda.empty_cache()
    
    # Load model and sampler from checkpoint
    model, sampler = load_model_and_sampler(checkpoint_path, modelConfig, device)
    
    # Generate profiles and store in a dict with keys: (energy, xy, z, material)
    gen_profiles_dict = {}
    for energy in energy_values_raw:
        mapped_energy = raw_energy_to_label[energy]
        for (xy_val, z_val) in cell_sizes:
            for material in material_conditions:
                key = (energy, xy_val, z_val, material)
                try:
                    profiles = get_generated_energy_profiles_for_labels(
                        sampler, modelConfig, device, mapped_energy, xy_val, z_val, material_value=material)
                    gen_profiles_dict[key] = profiles
                    print(f"Computed generated profiles for energy {energy} GeV, xy {xy_val}, z {z_val}, material {material}.")
                except Exception as e:
                    print(f"Skipping key {key}: {e}")
    
    # Now plot profiles for each cell size and for both materials
    save_directory = "./evaluation_results/"
    plot_profiles_for_cell_sizes(real_profiles_dict, gen_profiles_dict, energy_values_raw, cell_sizes, material_mapping, save_dir=save_directory)
