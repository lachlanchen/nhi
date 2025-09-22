import os
import torch
import numpy as np
import tifffile
from cellpose import models, io
from cellpose.io import imread

def read_pt_tensor(pt_path):
    tensor = torch.load(pt_path)
    return tensor.cpu().numpy()

def save_as_tiff(data, save_path):
    tifffile.imwrite(save_path, data)

def apply_cellpose_segmentation(tensor, model_type='cyto3', diameter=None, channels=[0,0], do_3D=True):
    model = models.Cellpose(model_type=model_type)
    masks, flows, styles, diams = model.eval(tensor, diameter=diameter, channels=channels, do_3D=do_3D)
    return masks

def main(pt_path, output_dir, model_type='cyto3', diameter=None, channels=[0,0], do_3D=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the tensor
    tensor = read_pt_tensor(pt_path)

    # Apply Cellpose segmentation
    # masks = apply_cellpose_segmentation(tensor, model_type=model_type, diameter=diameter, channels=channels, do_3D=do_3D)
    
    # Save the masks as TIFF
    base_name = os.path.splitext(os.path.basename(pt_path))[0]
    # save_path = os.path.join(output_dir, f"{base_name}_segmented.tiff")
    tensor_path = os.path.join(output_dir, f"{base_name}_original.tiff")
    # save_as_tiff(masks, save_path)
    save_as_tiff(tensor[:, 300:700, 500:1100], tensor_path)
    
    # print(f"Segmented TIFF saved at: {save_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment tensor file using Cellpose and save as TIFF')
    parser.add_argument('pt_path', type=str, help='Path to the .pt tensor file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the segmented TIFF file')
    parser.add_argument('--model_type', type=str, default='cyto3', help='Model type for Cellpose segmentation')
    parser.add_argument('--diameter', type=float, default=None, help='Diameter of the cells for Cellpose segmentation')
    parser.add_argument('--channels', type=int, nargs=2, default=[0,0], help='Channels for Cellpose segmentation')
    parser.add_argument('--do_3D', action='store_true', help='Whether to run 3D segmentation')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.pt_path)
    
    main(args.pt_path, args.output_dir, model_type=args.model_type, diameter=args.diameter, channels=args.channels, do_3D=args.do_3D)
