
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eig
from pymatting.util.util import row_sum
from scipy.sparse import diags
from sklearn.decomposition import PCA
import math

def get_grad_cam(grads, cams, modality, img_len=None, text_len=None):
    """Compute Grad-CAM for image or text tokens.

    The attention maps and gradients may come either in ``(B, H, N, N)`` format
    or ``(H, N, N)`` if the batch dimension was stripped by hooks.  This helper
    handles both representations.
    """

    final_gradcam = []
    num_layers = len(cams)
    for i in range(num_layers):
        grad = grads[i]
        cam = cams[i]

        if grad.dim() == 4:
            grad = grad[0]
        if cam.dim() == 4:
            cam = cam[0]

        if modality == "image":
            if img_len is None:
                raise ValueError("img_len must be provided for image modality")
            cam = cam[:, 1:img_len + 1, 1:img_len + 1]
            grad = grad[:, 1:img_len + 1, 1:img_len + 1].clamp(0)
        elif modality == "text":
            if img_len is None or text_len is None:
                raise ValueError("img_len and text_len must be provided for text modality")
            start = img_len + 1
            end = start + text_len
            cam = cam[:, start:end, start:end]
            grad = grad[:, start:end, start:end].clamp(0)
        else:
            print("Invalid modality")
            return None

        # Multiply gradients by attention maps (element-wise)
        layer_gradcam = cam * grad

        # Average over the attention heads to get a single map per layer
        layer_gradcam = layer_gradcam.mean(1)

        final_gradcam.append(layer_gradcam.cpu())
        
        
    if modality == "image":
        final_gradcam_np = torch.stack(final_gradcam).cpu().detach().numpy()
        final_gradcam_temp = np.mean(final_gradcam_np, axis=0)
        gradcam = final_gradcam_temp
        heatmap = np.mean(gradcam, axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap.flatten(), final_gradcam
        
        
    elif modality == "text":
        final_gradcam_np = torch.stack(final_gradcam).cpu().detach().numpy()
        final_gradcam_temp = np.mean(final_gradcam_np, axis=0)
        text_relearance = np.mean(final_gradcam_temp, axis=0)
        text_relearance = (text_relearance - text_relearance.min()) / (text_relearance.max() - text_relearance.min() + 1e-8)

        return text_relearance, final_gradcam


def get_rollout(cams, modality, img_len=None, text_len=None, eps=1e-8):
    """Attention rollout across layers for the given modality, with debug/info prints."""
    num_layers = len(cams)
    x = None

    for i, cam in enumerate(cams):
        # drop the batch dim if present
        if cam.dim() == 4:
            cam = cam[0]

        # slice out the relevant block
        if modality == "image":
            if img_len is None:
                raise ValueError("img_len must be provided for image modality")
            cam_i = cam[:, 1:img_len+1, 1:img_len+1]
        elif modality == "text":
            if img_len is None or text_len is None:
                raise ValueError("img_len and text_len must be provided for text modality")
            start = img_len + 1
            end = start + text_len
            cam_i = cam[:, start:end, start:end]
        else:
            # print(f"[get_rollout] Invalid modality: {modality!r}")
            return None

        # average over attention heads
        cam_i_avg = cam_i.mean(dim=0)

        # debug: check for NaNs in this layer’s average attention
        if torch.isnan(cam_i_avg).any():
            n_nan = torch.isnan(cam_i_avg).sum().item()
            # print(f"[Layer {i}] WARNING: cam_i_avg contains {n_nan} NaNs; replacing with zeros")
            cam_i_avg = torch.nan_to_num(cam_i_avg, nan=0.0, posinf=0.0, neginf=0.0)

        # initialize or multiply into the rollout
        if x is None:
            x = cam_i_avg.clone()
        else:
            # debug: show a quick snapshot of x before multiplication
            # print(f"[Layer {i}] x.norm before multiply = {x.norm(p=2).item():.4f}")
            x = x * cam_i_avg

            # debug: check for NaNs after multiply
            if torch.isnan(x).any():
                n_nan = torch.isnan(x).sum().item()
                # print(f"[Layer {i}] WARNING: x contains {n_nan} NaNs after multiplication; zeroing those")
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # safe normalization
            norm_x = x.norm(p=2)
            if norm_x > eps:
                x = x / (norm_x + eps)
            else:
                # print(f"[Layer {i}] INFO: norm too small ({norm_x.item():.4e}), skipping normalization")
                pass

    # move to CPU/NumPy and flatten
    heatmap = x.cpu().detach().numpy()
    heatmap = heatmap.mean(axis=0)
    # final normalization
    min_, max_ = heatmap.min(), heatmap.max()
    heatmap = (heatmap - min_) / (max_ - min_ + eps)

    return heatmap.flatten()


def get_diagonal (W):
    D = row_sum(W)
    D[D < 1e-12] = 1.0  # Prevent division by zero.
    D = diags(D)
    return D

def get_eigs (feats, modality, how_many = None, device="cpu"):
    if feats.size(0) == 1:
        feats = feats.detach().squeeze()


    if modality == "image":
        n_image_feats = feats.size(0)
        val = int( math.sqrt(n_image_feats) )
        if val * val == n_image_feats:
            feats = F.normalize(feats, p = 2, dim = -1).to(device)
        elif val * val + 1 == n_image_feats:
            feats = F.normalize(feats, p = 2, dim = -1)[1:].to(device)
        else:
            print(f"Invalid number of features detected: {n_image_feats}")

    else:
        feats = F.normalize(feats, p = 2, dim = -1)[1:-1].to(device)

    W_feat = (feats @ feats.T)
    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max() 

    W_feat = W_feat.detach().cpu().numpy()

    
    D = np.array(get_diagonal(W_feat).todense())

    L = D - W_feat

    L_shape = L.shape[0]
    if how_many >= L_shape - 1: 
        how_many = L_shape - 2

    try:
        eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5, M = D)
    except:
        try:
            eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5)
        except:
            eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM')
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
    
    n_tuple = torch.kthvalue(eigenvalues.real, 2)
    fev_idx = n_tuple.indices
    fev = eigenvectors[fev_idx].to(device)
        
    fev = torch.abs(fev)
    fevs_final = (fev - fev.min()) / (fev.max() - fev.min() + 1e-8)

    return fevs_final

def eig_seed(feats, modality, iters)        :
    patch_scores_norm = get_eigs(feats, modality, how_many = 5)
    num_patches = int(np.sqrt(len(patch_scores_norm)))
    heatmap = patch_scores_norm.reshape(num_patches, num_patches)  # Shape: [num_patches, num_patches]

    
    seed_index = np.argmax(patch_scores_norm)

    # Convert the 1D index to 2D indices
    seed_row = seed_index // num_patches
    seed_col = seed_index % num_patches


    # Initialize a mask for the expanded seed region
    seed_mask = np.zeros_like(heatmap)
    seed_mask[seed_row, seed_col] = 1

    # Define the number of expansion iterations
    num_expansion_iters = iters

    # Perform seed expansion
    for _ in range(num_expansion_iters):
        # Find neighboring patches
        neighbor_mask = cv2.dilate(seed_mask, np.ones((3, 3), np.uint8), iterations=1)
        neighbor_mask = neighbor_mask - seed_mask  # Exclude already included patches
        neighbor_indices = np.where(neighbor_mask > 0)
        
        # For each neighbor, decide whether to include it based on similarity
        for r, c in zip(*neighbor_indices):
            # Use heatmap values as similarity scores
            similarity = heatmap[r, c]
            # Define a threshold for inclusion
            threshold = 0.5  # Adjust this value as needed
            
            if similarity >= threshold:
                seed_mask[r, c] = 1  # Include the neighbor
            else:
                seed_mask[r, c] = 0.001

    # Apply the seed mask to the heatmap
    refined_heatmap = heatmap * seed_mask
    
    return refined_heatmap.flatten()
    
def get_pca_component(feats, modality, component=0, device="cpu"):
    if feats.size(0) == 1:
        feats = feats.detach().squeeze()
    original_len = feats.size(0)

    if modality == "image":
        n_image_feats = feats.size(0)
        val = int(math.sqrt(n_image_feats))
        if val * val == n_image_feats:
            feats = F.normalize(feats, p=2, dim=-1).to(device)
        elif val * val + 1 == n_image_feats:
            feats = F.normalize(feats, p=2, dim=-1)[1:].to(device)
        else:
            print(f"Invalid number of features detected: {n_image_feats}")
    else:
        feats = F.normalize(feats, p=2, dim=-1)[1:-1].to(device)

    # Reshape features to apply PCA on a [num_patches, feature_dim] matrix
    feats_reshaped = feats.cpu().detach().numpy()

    # Apply PCA on the reshaped data to get the second principal component
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(feats_reshaped)

    # Extract the second principal component and expand to original shape
    second_pc = principal_components[:, component]

    # Convert to tensor and move to the specified device
    second_pc = torch.tensor(second_pc, dtype=torch.float32).to(device)


    second_pc = torch.abs(second_pc)
    
    # Normalize the second principal component for visualization
    second_pc_norm = (second_pc - second_pc.min()) / (second_pc.max() - second_pc.min() + 1e-8)

    if modality == "text" and second_pc_norm.numel() == original_len - 2:
        # Pad to recover the original token length
        second_pc_norm = F.pad(second_pc_norm, (1, 1))

    return second_pc_norm

def get_image_relevance(ret, grads, cams):
    dsm = get_eigs(ret['image_feats'], "image", how_many = 5)
    lost = eig_seed(ret['image_feats'], "image", 15)
    pca_0 = get_pca_component(ret['image_feats'], "image", 0)

    dsm = np.array(dsm)
    lost = np.array(lost)
    pca_0 = np.array(pca_0)
    
    x = np.array([dsm, lost, pca_0])
    x = np.sum(x,axis=0)
    
    image_len = ret['image_feats'].shape[0] if ret['image_feats'].dim() == 2 else ret['image_feats'].shape[1]
    grad_cam, _ = get_grad_cam(grads, cams, "image", img_len=image_len)
    grad_cam = grad_cam * 2
    rollout = get_rollout(cams, "image", img_len=image_len)
    pca_1 = get_pca_component(ret['image_feats'], "image", 1)
    pca_1 = np.array(pca_1) * 0.001
    
    y = np.array([grad_cam,rollout, pca_1])
    
    y = np.sum(y,axis=0)
    
    z = x + y
    
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    
    return z


def get_text_relevance(ret, grads, cam):
    pca_0 = get_pca_component(ret['text_feats'], "text", 0)
    pca_0 = np.array(pca_0)
    
    x = np.array([pca_0])
    x = np.sum(x,axis=0)
    
    image_len = ret['image_feats'].shape[0] if ret['image_feats'].dim() == 2 else ret['image_feats'].shape[1]
    text_len = ret['text_feats'].shape[0] if ret['text_feats'].dim() == 2 else ret['text_feats'].shape[1]
    grad_cam, _ = get_grad_cam(grads, cam, "text", img_len=image_len, text_len=text_len)
    rollout = get_rollout(cam, "text", img_len=image_len, text_len=text_len)
    pca_1 = get_pca_component(ret['text_feats'], "text", 1)
    pca_1 = np.array(pca_1) * 0.01
    
    y = np.array([grad_cam,rollout, pca_1])
    y = np.sum(y,axis=0)
    
    z = x + y
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    
    return z
