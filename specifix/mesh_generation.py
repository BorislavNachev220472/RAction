# This script generates 3D meshes as STL files from a NIfTI file, which is the output mask from the nnUnet model.
# The script follows four main steps:
# 1. Converts the NIfTI file to label-specific NRRD files, where each label is saved as a separate binary mask.
# 2. Converts each NRRD file into a folder of TIFF slices.
# 3. Processes the TIFF slices into 3D mesh models using VTK, and generates STL files for each label.
# 4. Renames and copies the STL files to the output directory, while removing any temporary files created during the process.
#
# The generated STL files can be used for 3D rendering or further analysis of medical image data.

import argparse
import shutil

from skimage import io
import glob
import vtk
import nibabel as nib
import numpy as np
import nrrd
import os
import time
import warnings


def process_tiff_to_stl(data_file_path, output_stl_path, dataspacing, render, label, reduction_factor=0.9):
    data_file_path = data_file_path
    output_stl_path = output_stl_path
    dataspacing = dataspacing
    render = render
    label = label

    tiff_files = sorted(glob.glob(os.path.join(data_file_path, "*.tiff")))
    if not tiff_files:
        raise RuntimeError(f"No TIFF files found in the directory: {data_file_path}")

    print(f"First few TIFF files: {tiff_files[:5]}")

    reader = vtk.vtkTIFFReader()
    num_slices = len(tiff_files)
    print(f"Number of slices: {num_slices}")

    reader.SetFilePrefix(data_file_path)
    reader.SetFilePattern("%s%05d.tiff")
    reader.SetDataExtent(0, 511, 0, 511, 1, num_slices)
    reader.SetDataOrigin(0, 0, 0)
    reader.SetDataScalarTypeToUnsignedShort()
    reader.Update()

    print(f"Reader output extent: {reader.GetOutput().GetExtent()}")

    changeFilter = vtk.vtkImageChangeInformation()
    changeFilter.SetInputConnection(reader.GetOutputPort())
    changeFilter.SetSpacingScale(dataspacing)
    changeFilter.Update()

    slicestack = changeFilter.GetOutput()
    print(f"Slicestack extent: {slicestack.GetExtent()}")
    print(f"Slicestack number of points: {slicestack.GetNumberOfPoints()}")

    if slicestack.GetNumberOfPoints() == 0:
        raise RuntimeError("Failed to get valid output from changeFilter. Check your input data and parameters.")

    # Apply Gaussian smoothing
    gaussian = vtk.vtkImageGaussianSmooth()
    gaussian.SetInputData(slicestack)
    gaussian.SetStandardDeviation(0.5)
    gaussian.Update()

    # Marching Cubes for initial surface extraction
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(gaussian.GetOutput())
    marching_cubes.SetValue(0, 0.5)
    marching_cubes.Update()

    complete_mesh = marching_cubes.GetOutput()

    # Subdivision for higher resolution mesh
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(complete_mesh)
    decimate.SetTargetReduction(reduction_factor)
    decimate.PreserveTopologyOn()
    decimate.Update()
    #
    high_res_mesh = decimate.GetOutput()

    # high_res_mesh = subdiv.GetOutput()

    # Apply advanced smoothing filter
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(high_res_mesh)
    smoother.SetNumberOfIterations(10)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(120.0)
    smoother.SetPassBand(0.001)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    smooth_mesh = smoother.GetOutput()

    # Clean the final mesh
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(smooth_mesh)
    cleaner.Update()

    refined_mesh = cleaner.GetOutput()

    connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
    connectivity_filter.SetInputData(refined_mesh)
    connectivity_filter.SetExtractionModeToAllRegions()
    connectivity_filter.ColorRegionsOn()
    connectivity_filter.Update()

    num_components = connectivity_filter.GetNumberOfExtractedRegions()
    num_points_complete = refined_mesh.GetNumberOfPoints()
    print(f"Number of components: {num_components}")
    print(f"Number of points in refined mesh: {num_points_complete}")

    if render:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(connectivity_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        renderer.AddActor(actor)
        interactor.Initialize()
        render_window.Render()
        interactor.Start()

    writer = vtk.vtkSTLWriter()
    components = []
    largest_n = 0
    largest_id = 0
    for i in range(num_components):
        connectivity_filter.InitializeSpecifiedRegionList()
        connectivity_filter.SetExtractionModeToSpecifiedRegions()
        connectivity_filter.AddSpecifiedRegion(i)
        connectivity_filter.Update()
        component = vtk.vtkPolyData()
        connectivity_filter.DeleteSpecifiedRegion(i)
        component.DeepCopy(connectivity_filter.GetOutput())

        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.SetInputData(component)
        cleanPolyData.Update()

        if cleanPolyData.GetOutput().GetNumberOfPoints() < num_points_complete * 0.1:
            continue

        components.append(cleanPolyData.GetOutput())

        if cleanPolyData.GetOutput().GetNumberOfPoints() > largest_n:
            largest_n = cleanPolyData.GetOutput().GetNumberOfPoints()
            largest_id = len(components) - 1

        if render:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(cleanPolyData.GetOutput())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            renderer = vtk.vtkRenderer()
            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)

            renderer.AddActor(actor)
            interactor.Initialize()
            render_window.Render()
            interactor.Start()

    print(f"Number of connected components: {len(components)}")

    for i, component in enumerate(components):
        writer.SetInputData(components[i])

        if label == "label_13":
            continue
        if label == "label_14":
            continue
        if label == "label_3":
            if i == largest_id:
                filename = "shaft.stl"
            else:
                filename = f"{label}_{str(i + 1).zfill(2)}.stl"
        else:
            filename = f"{label}_{str(i + 1).zfill(2)}.stl"

        full_path = os.path.join(output_stl_path, filename)
        writer.SetFileName(full_path)

        print(writer.GetFileName())
        writer.Write()
        print(
            f"Connected component {i + 1}: Number of points = {component.GetNumberOfPoints()}, Number of cells = {component.GetNumberOfCells()}")


def nifti_to_label_nrrds(nifti_filename, output_folder):
    """
    Converts a NIfTI file to multiple NRRD files based on different labels found within the NIfTI file.

    Parameters:
    nifti_filename (str): Full path to the NIfTI file that contains the segmented medical image data.
    output_folder (str): Directory where the NRRD files for each label will be saved. It will be created if it does not exist.

    The function reads the NIfTI file, identifies all unique labels in the data, and generates a binary mask for each label.
    Each mask is saved as a separate NRRD file named according to its label.
    """
    # Load the NIfTI file
    nifti_img = nib.load(nifti_filename)
    header = nifti_img.header
    spacing = header.get_zooms()[:3]
    data = nifti_img.get_fdata()

    # Get unique labels, excluding the background (0)
    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels > 0]

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate and save a NRRD file for each label
    for label in unique_labels:
        label_mask = (data == label).astype(np.uint8)  # Create a binary mask for the current label
        label_mask[label_mask > 0] = 1  # Ensure the mask is binary

        label_nrrd_filename = os.path.join(output_folder, f"label_{int(label)}.nrrd")
        nrrd.write(label_nrrd_filename, label_mask)
        print(f"Saved NRRD file for label {label} to {label_nrrd_filename}")
    return spacing


def convert_nrrd_to_tiff_slices(nrrd_file, output_folder):
    """
    Converts a single NRRD file to a series of TIFF slice files. Each slice of the NRRD file is saved as a separate TIFF file
    in a newly created folder named after the NRRD file.

    Parameters:
    nrrd_file (str): Path to the NRRD file.
    output_folder (str): Base directory where the folders containing TIFF files will be created.
    """
    # Read the NRRD file
    data, _ = nrrd.read(nrrd_file)

    # Create output directory for this NRRD file's TIFFs
    nrrd_base_name = os.path.basename(nrrd_file).replace('.nrrd', '')
    tiff_folder = os.path.join(output_folder, nrrd_base_name)
    os.makedirs(tiff_folder, exist_ok=True)

    # Save each slice of the array as a TIFF image
    for i in range(data.shape[0]):  # Assuming the third dimension is the slice dimension
        slice_filename = os.path.join(tiff_folder, f"{str(i + 1).zfill(5)}.tiff")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(slice_filename, data[i, :, :].astype(np.float32))
        # print(f"Saved slice {i + 1} as TIFF: {slice_filename}")


def rename_and_copy_stl_files(input_folder, output_folder):
    stl_files = [f for f in os.listdir(input_folder) if f.endswith('.stl')]

    # Delete files smaller than 50 KB
    for filename in stl_files[:]:
        file_path = os.path.join(input_folder, filename)
        if os.path.getsize(file_path) < 100 * 1024:
            os.remove(file_path)
            print(f"Deleted file {file_path} because it is smaller than 100 KB")
            stl_files.remove(filename)

    stl_files.sort()  # Ensure consistent ordering

    os.makedirs(output_folder, exist_ok=True)

    counter = 1
    for filename in stl_files:
        src = os.path.join(input_folder, filename)
        if filename == "shaft.stl":
            dst = os.path.join(output_folder, filename)
        else:
            dst = os.path.join(output_folder, f"frag_{counter}.stl")
            counter += 1
        shutil.copyfile(src, dst)
        print(f"Copied {src} to {dst}")


if __name__ == "__main__":
    # Set up a parser for command line arguments
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Convert NIfTI files to label-specific NRRD files. Each label from the NIfTI file is saved as a separate binary NRRD file.")
    parser.add_argument("nifti_filename", type=str, help="Path to the NIfTI file containing the medical image data.")
    parser.add_argument("output_folder", type=str, help="Directory where the NRRD files will be saved.")
    args = parser.parse_args()

    ##### STEP 1,     # Convert NIfTI to label NRRDs using the provided arguments
    tmp_nrrd_folder = "/tmp_nrrd_folder"
    os.makedirs(tmp_nrrd_folder, exist_ok=True)
    spacing = nifti_to_label_nrrds(args.nifti_filename, tmp_nrrd_folder)

    #### STEP 2: convert each nrrd to a separate tiff folder
    tmp_tiff_folder = "/tmp_tiff_folder"
    # if not os.path.isdir(tmp_tiff_folder):
    os.makedirs(tmp_tiff_folder, exist_ok=True)

    # Find all NRRD files in the input directory
    nrrd_files = glob.glob(os.path.join(tmp_nrrd_folder, "*.nrrd"))
    if not nrrd_files:
        print("No NRRD files found.")
        exit()

    # Convert each NRRD file to a folder of TIFF slices
    for nrrd_file in nrrd_files:
        convert_nrrd_to_tiff_slices(nrrd_file, tmp_tiff_folder)

    #### STEP 3: convert each tiff folder to one stl

    tmp_stl_folder = "./tmp_stl_folder"
    os.makedirs(tmp_stl_folder, exist_ok=True)

    for subdir in os.scandir(tmp_tiff_folder):
        if subdir.is_dir():
            label = os.path.basename(subdir.path)
            path = subdir.path
            if not path.endswith(os.sep):
                path += os.sep
            print(f"Processing subfolder: {label}")
            print(f"Processing subfolder: {subdir.path}")
            process_tiff_to_stl(
                data_file_path=path,
                output_stl_path=tmp_stl_folder,
                dataspacing=spacing,
                render=False,
                label=label
            )
    #### STEP 4: rename and copy fragments to the output directory
    rename_and_copy_stl_files(tmp_stl_folder, args.output_folder)
    # and delete the folders
    if os.path.exists(tmp_nrrd_folder):
        shutil.rmtree(tmp_nrrd_folder)
    if os.path.exists(tmp_stl_folder):
        shutil.rmtree(tmp_stl_folder)
    if os.path.exists(tmp_tiff_folder):
        shutil.rmtree(tmp_tiff_folder)
    elapsed_time = time.perf_counter() - start_time
    print(f"The script took {elapsed_time:.2f} seconds to run.")
