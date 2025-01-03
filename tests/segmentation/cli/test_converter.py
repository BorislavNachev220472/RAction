import numpy as np
import SimpleITK as sitk
import pytest

from specifix.segmentation.cli.converter import Converter


class TestStackNrrd:

    @staticmethod
    def create_static_image(size, value=0):
        arr = np.full(size, value, dtype=np.int8)
        return sitk.GetImageFromArray(arr)

    @pytest.fixture
    def obj(self, mocker):
        size = (10, 10, 10)
        mock_image = self.create_static_image(size)
        mock_read_image = mocker.patch("SimpleITK.ReadImage", return_value=mock_image)
        return Converter('mock_image')

    def test_stack_single_image(self, obj):
        size = (10, 10, 10)
        label = 1
        new_image = self.create_static_image(size, value=1)

        result = obj.stack_nrrd(label, None, new_image)

        # Create expected result
        expected = self.create_static_image(size, value=int(label))
        assert np.array_equal(sitk.GetArrayFromImage(result), sitk.GetArrayFromImage(expected))

    def test_stack_two_images_same_size(self, obj):
        size = (10, 10, 10)
        prev_image = self.create_static_image(size, value=1)
        new_image = self.create_static_image(size, value=2)

        result = obj.stack_nrrd(None, prev_image, new_image)

        new_image_labeled = sitk.Cast(new_image, sitk.sitkInt8)
        expected = sitk.Maximum(prev_image, new_image_labeled)
        assert np.array_equal(sitk.GetArrayFromImage(result), sitk.GetArrayFromImage(expected))

    def test_stack_two_images_different_size(self, obj):
        size1 = (10, 10, 10)
        size2 = (15, 15, 15)
        label = 3
        prev_image = self.create_static_image(size1, value=1)
        new_image = self.create_static_image(size2, value=2)

        result = obj.stack_nrrd(label, prev_image, new_image)

        assert result.GetSize() == prev_image.GetSize()

    def test_stack_with_label(self, obj):
        size = (10, 10, 10)
        label = 5
        prev_image = self.create_static_image(size, value=1)
        new_image = self.create_static_image(size, value=3)

        result = obj.stack_nrrd(label, prev_image, new_image)

        expected = self.create_static_image(size, value=label)
        assert np.array_equal(sitk.GetArrayFromImage(result), sitk.GetArrayFromImage(expected))


class TestCopyMetadataFromImage:

    def create_test_image(self, size, value=1, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        image = sitk.Image(size, sitk.sitkUInt8)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        image += value
        return image

    @pytest.fixture
    def obj(self, mocker):
        size = (10, 10, 10)
        mock_image = self.create_test_image(size)
        mock_read_image = mocker.patch("SimpleITK.ReadImage", return_value=mock_image)
        return Converter('mock_image')

    def test_no_filtering_data(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=1)
        result_image = obj.copy_metadata_from_image(original_image)
        assert np.array_equal(sitk.GetArrayFromImage(result_image), sitk.GetArrayFromImage(original_image))

    def test_no_filtering_spacing(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=1)
        result_image = obj.copy_metadata_from_image(original_image)
        assert result_image.GetSpacing() == original_image.GetSpacing()

    def test_no_filtering_origin(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=1)
        result_image = obj.copy_metadata_from_image(original_image)
        assert result_image.GetOrigin() == original_image.GetOrigin()

    def test_with_filtering_data(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=1)
        label = 1
        labeled_image = obj.copy_metadata_from_image(original_image, label=label)
        expected_array = (sitk.GetArrayFromImage(original_image) == label).astype(np.uint8)
        assert np.array_equal(sitk.GetArrayFromImage(labeled_image), expected_array)

    def test_with_filtering_spacing(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=1)
        label = 1
        labeled_image = obj.copy_metadata_from_image(original_image, label=label)
        assert labeled_image.GetSpacing() == original_image.GetSpacing()

    def test_with_filtering_origin(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=1)
        label = 1
        labeled_image = obj.copy_metadata_from_image(original_image, label=label)
        assert labeled_image.GetOrigin() == original_image.GetOrigin()


class TestCopyMetadataFromArray:

    def create_test_image(self, size, value=1, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        image = sitk.Image(size, sitk.sitkUInt8)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        image += value
        return image

    @pytest.fixture
    def obj(self, mocker):
        size = (10, 10, 10)
        mock_image = self.create_test_image(size)
        mock_read_image = mocker.patch("SimpleITK.ReadImage", return_value=mock_image)
        return Converter('mock_image')

    def test_no_filtering_data(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=2)
        array = sitk.GetArrayFromImage(original_image)
        result_image = obj.copy_metadata_from_array(original_image, array)
        assert np.array_equal(sitk.GetArrayFromImage(result_image), array)

    def test_no_filtering_spacing(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=2)
        array = sitk.GetArrayFromImage(original_image)
        result_image = obj.copy_metadata_from_array(original_image, array)
        assert result_image.GetSpacing() == original_image.GetSpacing()

    def test_no_filtering_origin(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=2)
        array = sitk.GetArrayFromImage(original_image)
        result_image = obj.copy_metadata_from_array(original_image, array)
        assert result_image.GetOrigin() == original_image.GetOrigin()

    def test_with_filtering_data(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=2)
        array = sitk.GetArrayFromImage(original_image)
        label = 2
        filtered_image = obj.copy_metadata_from_array(original_image, array, label=label)
        expected_array = (array == label).astype(np.uint8)
        assert np.array_equal(sitk.GetArrayFromImage(filtered_image), expected_array)

    def test_with_filtering_spacing(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=2)
        array = sitk.GetArrayFromImage(original_image)
        label = 2
        filtered_image = obj.copy_metadata_from_array(original_image, array, label=label)
        assert filtered_image.GetSpacing() == original_image.GetSpacing()

    def test_with_filtering_origin(self, obj):
        size = (5, 5, 5)
        original_image = self.create_test_image(size=size, value=2)
        array = sitk.GetArrayFromImage(original_image)
        label = 2
        filtered_image = obj.copy_metadata_from_array(original_image, array, label=label)
        assert filtered_image.GetOrigin() == original_image.GetOrigin()


class TestMetadataOperations:

    def create_test_image(self, size, value=1, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        image = sitk.Image(size, sitk.sitkUInt8)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        image += value
        return image

    @pytest.fixture
    def obj(self, mocker):
        size = (10, 10, 10)
        mock_image = self.create_test_image(size)
        mock_read_image = mocker.patch("SimpleITK.ReadImage", return_value=mock_image)
        return Converter('mock_image')

    def test_get_array_from_image(self, obj):
        size = (5, 5, 5)
        image = self.create_test_image(size=size, value=3)

        result_array = obj.get_array_from_image(image)
        expected_array = sitk.GetArrayFromImage(image)
        assert np.array_equal(result_array, expected_array)

    def test_get_image_from_array(self, obj):
        array = np.ones((5, 5, 5), dtype=np.uint8)

        result_image = obj.get_image_from_array(array)
        expected_image = sitk.GetImageFromArray(array)
        assert np.array_equal(sitk.GetArrayFromImage(result_image), sitk.GetArrayFromImage(expected_image))
