import os
from datetime import datetime, timezone

import vtk
import time
import numpy as np
import SimpleITK as sitk

from vtkmodules.util import numpy_support
from concurrent.futures import ThreadPoolExecutor

from specifix.segmentation.cli.config import Config
from specifix.segmentation.cli.converter import Converter
from specifix.segmentation.cli.io import BoneIOFileManager


class Processor(object):
    """
    A utility class for processing VTK polydata using various filters and techniques for smoothing, resolution enhancement,
    cleaning, and connectivity analysis.
    """

    def __init__(self):
        """
        Initializes the Processor object.
        """
        pass

    def apply_gausian_smooting(self, poly_data: vtk.vtkPolyData, standard_deviation=0.5):
        """
        Applies Gaussian smoothing to the input polydata.

        Parameters:
            poly_data (vtk.vtkPolyData): The input polydata to smooth.
            standard_deviation (float): Standard deviation for Gaussian smoothing. Default is 0.5.

        Returns:
            vtk.vtkPolyData: Smoothed polydata.
        """
        gaussian = vtk.vtkImageGaussianSmooth()
        gaussian.SetInputData(poly_data)
        gaussian.SetStandardDeviation(standard_deviation)
        gaussian.Update()
        return gaussian.GetOutput()

    def apply_marshing_cubes(self, poly_data: vtk.vtkPolyData, index_isosurface=0, value_isosurface=0.5):
        """
        Extracts a surface using the Marching Cubes algorithm.

        Parameters:
            poly_data (vtk.vtkPolyData): The input polydata.
            index_isosurface (int): The isosurface index. Default is 0.
            value_isosurface (float): The isosurface value. Default is 0.5.

        Returns:
            vtk.vtkPolyData: The extracted isosurface polydata.
        """
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(poly_data)
        marching_cubes.SetValue(index_isosurface, value_isosurface)
        marching_cubes.Update()
        return marching_cubes.GetOutput()

    def apply_subdivision_for_higher_resolution(self, poly_data: vtk.vtkPolyData, reduction_factor: float):
        """
        Reduces the resolution of the polydata by applying a subdivision filter.

        Parameters:
            poly_data (vtk.vtkPolyData): The input polydata.
            reduction_factor (float): The target reduction factor (0.0 to 1.0). Higher values reduce more.

        Returns:
            vtk.vtkPolyData: The resolution-reduced polydata.
        """
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(poly_data)
        decimate.SetTargetReduction(reduction_factor)
        decimate.PreserveTopologyOn()
        decimate.Update()
        return decimate.GetOutput()

    def apply_advanced_smooting_filter(self, poly_data: vtk.vtkPolyData, iterations: int = 10,
                                       feature_angle: float = 120.0, pass_band: float = 0.001):
        """
        Applies advanced smoothing using the Windowed Sinc PolyData Filter.

        Parameters:
            poly_data (vtk.vtkPolyData): The input polydata to smooth.
            iterations (int): Number of smoothing iterations. Default is 10.
            feature_angle (float): Feature angle to preserve edges. Default is 120.0 degrees.
            pass_band (float): Pass band value for smoothing. Default is 0.001.

        Returns:
            vtk.vtkPolyData: Smoothed polydata.
        """
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(poly_data)
        smoother.SetNumberOfIterations(iterations)
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(feature_angle)
        smoother.SetPassBand(pass_band)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        return smoother.GetOutput()

    def apply_cleaning(self, poly_data: vtk.vtkPolyData):
        """
        Cleans the input polydata by merging duplicate points and removing unused points.

        Parameters:
            poly_data (vtk.vtkPolyData): The input polydata to clean.

        Returns:
            vtk.vtkPolyData: Cleaned polydata.
        """
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(poly_data)
        cleaner.Update()
        return cleaner.GetOutput()

    def apply_connectivity_filter(self, poly_data: vtk.vtkPolyData):
        """
        Applies a connectivity filter to the polydata to extract connected regions.

        Parameters:
            poly_data (vtk.vtkPolyData): The input polydata.

        Returns:
            vtk.vtkPolyData: Polydata with connected regions identified.
        """
        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputData(poly_data)
        connectivity_filter.SetExtractionModeToAllRegions()
        connectivity_filter.InitializeSpecifiedRegionList()
        connectivity_filter.ColorRegionsOn()
        connectivity_filter.Update()
        return connectivity_filter.GetOutput()


class Mesh(BoneIOFileManager):

    def __init__(self, input_directory: str, output_directory: str, reference_filename: str):
        super().__init__(input_directory, output_directory)
        self.Converter = Converter(reference_filename)
        self.Processor = Processor()

    def process_image_to_stl(self, image: sitk.Image, label: int, reduction_factor: float = 0.9):
        """
        This function generates the mesh based on the input image.
        :param image: sitk.Image The image that contains the data tha
        :param label: int The label that should be used to map the label value to its text representation.
        :param reduction_factor: float The reduction factor that should be used when generating the mesh.
        :return: None
        """
        print(f"Number of slices: {label}")
        np_image = self.Converter.get_array_from_image(image)

        np_image = np_image.astype(np.uint16)
        num_slices, height, width = np_image.shape
        vtk_data_array = numpy_support.numpy_to_vtk(np_image.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)

        reader = vtk.vtkImageData()
        reader.SetDimensions(np_image.shape[2], np_image.shape[1], num_slices)
        reader.GetPointData().SetScalars(vtk_data_array)
        reader.SetExtent(0, np_image.shape[2] - 1, 0, np_image.shape[1] - 1, 0, num_slices - 1)
        reader.SetOrigin(0, 0, 0)
        reader.Modified()

        changeFilter = vtk.vtkImageChangeInformation()
        changeFilter.SetInputData(reader)
        print(f"DataSpacing: {image.GetSpacing()[::-1]}")
        changeFilter.SetSpacingScale(image.GetSpacing()[::-1])
        # changeFilter.SetSpacingScale(0.6, 0.3, 0.3)
        # changeFilter.SetSpacingScale(0.3, 0.3, 0.6)
        changeFilter.Update()

        slicestack = changeFilter.GetOutput()

        if slicestack.GetNumberOfPoints() == 0:
            raise RuntimeError("Failed to get valid output from changeFilter. Check your input data and parameters.")

        gaussian_output = self.Processor.apply_gausian_smooting(slicestack, standard_deviation=0.5)

        marching_cubes_output = self.Processor.apply_marshing_cubes(gaussian_output,
                                                                    index_isosurface=0,
                                                                    value_isosurface=0.8)

        decimate_output = self.Processor.apply_subdivision_for_higher_resolution(marching_cubes_output,
                                                                                 reduction_factor)

        advanced_smooting_output = self.Processor.apply_advanced_smooting_filter(decimate_output,
                                                                                 iterations=10,
                                                                                 feature_angle=120.0,
                                                                                 pass_band=0.001)

        cleaned_output = self.Processor.apply_cleaning(advanced_smooting_output)

        connectivity_output = self.Processor.apply_connectivity_filter(cleaned_output)

        component = vtk.vtkPolyData()
        component.DeepCopy(connectivity_output)
        cleanPolyData = self.Processor.apply_cleaning(component)
        # restore index from label (as it starts from 0)
        full_path = os.path.join(self.output_directory,
                                 f'{Config.DATA_STRUCTURE[label - 1].label}.stl')
        if os.path.exists(full_path):
            print('File already exists')
            tokens = os.path.basename(full_path).split('.')
            full_path = full_path.replace(tokens[0],
                                          f'{tokens[0]}_{datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")}')
        writer = vtk.vtkSTLWriter()
        writer.SetInputData(cleanPolyData)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        writer.SetFileName(full_path)

        print(writer.GetFileName())
        writer.Write()

    def generate_mesh(self, nifti_filename: str):
        """
        This function generates the meshes from a nifti file. The amount of meshes depends on the amount of unque labels
        of the nifti file. It uses Threads to run the genration process in parallel and speed up the execution time.
        :param nifti_filename: str the name of the nifti file.
        :return: None
        """
        total_start_time = time.perf_counter()

        #### STEP 1: convert nifti to a separate tiff folder for each label
        image = self.read_file(nifti_filename)

        label_array = self.Converter.get_array_from_image(image)
        unique_labels = np.unique(label_array)
        unique_labels = unique_labels[unique_labels != 0]
        print(f"Unique labels found: {unique_labels}")
        #### STEP 2: convert each tiff folder to one stl
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_image_to_stl,
                                       self.Converter.copy_metadata_from_array(image, label_array,
                                                                               label=label),
                                       label, 0.9) for label in unique_labels]

            for future in futures:
                future.result()

        total_elapsed_time = time.perf_counter() - total_start_time
        print(f"The script took {total_elapsed_time:.2f} seconds to run.")

    def generate_meshes(self, ct_scan: str):
        """
        This function generates the meshes of multiple nrrd files for a single CT scan stored in the following way.
        dir
            - CT_SCAN_NAME
                - nrrd
                    - *files*
        :param ct_scan: str The name of the CT scan.
        :return: None
        """
        total_start_time = time.perf_counter()

        current_files_iterator = iter((f for f in self.set_inner_folder(ct_scan)
                                      .set_inner_folder(Config.NRRD_INNER_DIRECTORY)
                                      .read_all() if f.endswith(Config.NRRD_EXTENSION)))
        for filename in current_files_iterator:
            self.generate_mesh(
                os.path.join(ct_scan, Config.NRRD_INNER_DIRECTORY, filename))

        total_elapsed_time = time.perf_counter() - total_start_time
        print(f"The script took {total_elapsed_time:.2f} seconds to run.")
