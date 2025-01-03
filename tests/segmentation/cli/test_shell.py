import pytest
import numpy as np
import SimpleITK as sitk
from specifix.segmentation.cli.shell import Shell


def create_test_image(size, value=1, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    image = sitk.Image(size, sitk.sitkUInt8)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image += value
    return image


@pytest.fixture
def shell(mocker):
    size = (10, 10, 10)
    mock_image = create_test_image(size)
    mock_read_image = mocker.patch("SimpleITK.ReadImage", return_value=mock_image)
    return Shell(input_directory="input", output_directory="output", reference_filename="reference")


@pytest.fixture
def mask():
    mask_array = np.zeros((10, 10), dtype=np.uint8)
    mask_array[3:7, 3:7] = 1  # Create a square in the middle
    return sitk.GetImageFromArray(mask_array)


@pytest.fixture
def mask_with_holes():
    mask_array = np.ones((10, 10), dtype=np.uint8)
    mask_array[3:7, 3:7] = 0  # Create a hole in the middle
    return sitk.GetImageFromArray(mask_array)


class TestPerformMorphologicalClosing:
    def test_perform_morphological_closing(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        closed_mask = shell.perform_morphological_closing(mask_array, iterations=3)
        assert np.any(closed_mask != mask_array)

    def test_perform_morphological_closing_edge_case(self, shell):
        empty_mask = sitk.GetImageFromArray(np.zeros((10, 10), dtype=np.uint8))
        empty_mask_array = sitk.GetArrayFromImage(empty_mask)
        closed_mask = shell.perform_morphological_closing(empty_mask_array, iterations=3)
        assert np.all(closed_mask == 0)

    def test_perform_morphological_closing_with_holes(self, shell, mask_with_holes):
        mask_array = sitk.GetArrayFromImage(mask_with_holes)
        closed_mask = shell.perform_morphological_closing(mask_array, iterations=3)
        assert np.all(closed_mask == 1)

    def test_perform_morphological_closing2(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        closed_mask = shell.perform_morphological_closing(mask_array, iterations=3)
        assert closed_mask.shape == mask_array.shape

    def test_perform_morphological_closing_edge_case2(self, shell):
        empty_mask = sitk.GetImageFromArray(np.zeros((10, 10), dtype=np.uint8))
        empty_mask_array = sitk.GetArrayFromImage(empty_mask)
        closed_mask = shell.perform_morphological_closing(empty_mask_array, iterations=3)
        assert closed_mask.shape == empty_mask_array.shape

    def test_perform_morphological_closing_with_holes2(self, shell, mask_with_holes):
        mask_array = sitk.GetArrayFromImage(mask_with_holes)
        closed_mask = shell.perform_morphological_closing(mask_array, iterations=3)
        assert closed_mask.shape == mask_array.shape


class TestThickenBoundary:
    def test_thicken_boundary(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        thickened_mask = shell.thicken_boundary(mask_array, iterations=2)
        assert np.any(thickened_mask != mask_array)

    def test_thicken_boundary_no_change(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        thickened_mask = shell.thicken_boundary(mask_array, iterations=0)
        assert np.all(thickened_mask == mask_array)

    def test_thicken_boundary_large_iterations(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        thickened_mask = shell.thicken_boundary(mask_array, iterations=5)
        assert np.any(thickened_mask != mask_array)

    def test_thicken_boundary2(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        thickened_mask = shell.thicken_boundary(mask_array, iterations=2)
        assert thickened_mask.shape == mask_array.shape

    def test_thicken_boundary_no_change2(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        thickened_mask = shell.thicken_boundary(mask_array, iterations=0)
        assert thickened_mask.shape == mask_array.shape

    def test_thicken_boundary_large_iterations2(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        thickened_mask = shell.thicken_boundary(mask_array, iterations=5)
        assert thickened_mask.shape == mask_array.shape


class TestRemoveInnerArtifacts:
    def test_remove_inner_artifacts(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        cleaned_mask = shell.remove_inner_artifacts(mask_array)
        assert cleaned_mask.shape == mask_array.shape

    def test_remove_inner_artifacts_empty(self, shell):
        empty_mask = sitk.GetImageFromArray(np.zeros((10, 10), dtype=np.uint8))
        cleaned_mask = shell.remove_inner_artifacts(empty_mask)
        assert np.all(cleaned_mask == 0)

    def test_remove_inner_artifacts2(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        cleaned_mask = shell.remove_inner_artifacts(mask_array)
        assert np.any(cleaned_mask == mask_array)

    def test_remove_inner_artifacts_empty2(self, shell):
        empty_mask = sitk.GetImageFromArray(np.zeros((10, 10), dtype=np.uint8))
        cleaned_mask = shell.remove_inner_artifacts(empty_mask)
        assert cleaned_mask.GetSize() == empty_mask.GetSize()


class TestExtractOuterBoundary:
    def test_extract_outer_boundary(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        outer_boundary = shell.extract_outer_boundary(mask_array)
        assert outer_boundary.shape == mask_array.shape

    def test_extract_outer_boundary2(self, shell, mask):
        mask_array = sitk.GetArrayFromImage(mask)
        outer_boundary = shell.extract_outer_boundary(mask_array)
        assert np.any(outer_boundary != mask_array)

    def test_extract_outer_boundary_with_holes(self, shell, mask_with_holes):
        mask_array = sitk.GetArrayFromImage(mask_with_holes)
        outer_boundary = shell.extract_outer_boundary(mask_array)
        assert outer_boundary.shape == mask_array.shape

    def test_extract_outer_boundary_with_holes2(self, shell, mask_with_holes):
        mask_array = sitk.GetArrayFromImage(mask_with_holes)
        outer_boundary = shell.extract_outer_boundary(mask_array)
        assert outer_boundary.shape == mask_array.shape
