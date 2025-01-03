import pytest
import vtk
from specifix.segmentation.cli.mesh import Processor


@pytest.fixture
def processor():
    return Processor()


@pytest.fixture
def poly_data():
    sphere = vtk.vtkSphereSource()
    sphere.Update()
    return sphere.GetOutput()


class TestApplyGaussianSmoothing:

    def test_apply_gaussian_smoothing(self, processor, poly_data):
        smoothed_data = processor.apply_gausian_smooting(poly_data, standard_deviation=0.5)
        assert smoothed_data is not None


class TestApplyMarchingCubes:

    def test_apply_marshing_cubes(self, processor, poly_data):
        isosurface_data = processor.apply_marshing_cubes(poly_data, index_isosurface=0, value_isosurface=0.5)
        assert isosurface_data is not None


class TestApplySubdivisionForHigherResolution:

    def test_apply_subdivision_for_higher_resolution(self, processor, poly_data):
        reduced_data = processor.apply_subdivision_for_higher_resolution(poly_data, reduction_factor=0.2)
        assert reduced_data is not None


class TestApplyAdvancedSmoothingFilter:

    def test_apply_advanced_smoothing_filter(self, processor, poly_data):
        smoothed_data = processor.apply_advanced_smooting_filter(poly_data, iterations=5, feature_angle=90,
                                                                 pass_band=0.01)
        assert smoothed_data is not None


class TestApplyCleaning:

    def test_apply_cleaning(self, processor, poly_data):
        cleaned_data = processor.apply_cleaning(poly_data)
        assert cleaned_data is not None


class TestApplyConnectivityFilter:

    def test_apply_connectivity_filter(self, processor, poly_data):
        connectivity_data = processor.apply_connectivity_filter(poly_data)
        assert connectivity_data is not None
