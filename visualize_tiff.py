import napari
import tifffile

def visualize_tiff(tiff_path):
    # Load the TIFF file
    data = tifffile.imread(tiff_path)
    
    # Create a napari viewer
    viewer = napari.Viewer()
    
    # Add the image data to the viewer
    viewer.add_image(data, name='Segmented Data', colormap='gray', blending='additive')
    
    # Start the napari event loop
    napari.run()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize 3D TIFF using napari')
    parser.add_argument('tiff_path', type=str, help='Path to the 3D TIFF file')
    
    args = parser.parse_args()
    
    visualize_tiff(args.tiff_path)
