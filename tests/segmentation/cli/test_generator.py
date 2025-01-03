import pytest
import SimpleITK as sitk

from specifix.segmentation.cli.generator import Generator
from specifix.segmentation.cli.config import Config


def create_test_image(size, value=1, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    image = sitk.Image(size, sitk.sitkInt8)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image += value
    return image


@pytest.fixture
def generator(mocker):
    size = (10, 10, 10)
    mock_image = create_test_image(size)
    mock_read_image = mocker.patch("SimpleITK.ReadImage", return_value=mock_image)
    return Generator(input_directory="input", output_directory="output", reference_filename="reference")


class TestAppendRecord:

    def test_unknown(self, generator: Generator):
        size = (10, 10, 10)
        f = create_test_image(size)
        s = create_test_image(size)
        result = generator.append_record('test_scan', 'test.nii.gz', Config.COMBINED_DICT, None, None)
        assert result is None

    def test_first_known(self, generator: Generator):
        size = (10, 10, 10)
        s = create_test_image(size)
        result = generator.append_record('test.nii.gz', Config.DATA_STRUCTURE[0].label, Config.COMBINED_DICT, None, s)
        assert result == sitk.Cast(s, sitk.sitkInt8)

    def test_second_known(self, generator: Generator):
        size = (10, 10, 10)
        f = create_test_image(size)
        s = create_test_image(size)
        result = generator.append_record('test.nii.gz', Config.DATA_STRUCTURE[0].label, Config.COMBINED_DICT, f, s)
        assert result == sitk.Cast(f, sitk.sitkInt8)

    def test_debug_dictionary(self, generator: Generator):
        size = (10, 10, 10)
        f = create_test_image(size)
        s = create_test_image(size)
        result = generator.append_record('test.nii.gz', Config.DATA_STRUCTURE[0].label, Config.COMBINED_DICT, f, s)
        assert generator.debug_dict['test.nii.gz'].count == 1

    def test_debug_dictionary_duplicates(self, generator: Generator):
        size = (10, 10, 10)
        f = create_test_image(size)
        s = create_test_image(size)
        result = generator.append_record('test.nii.gz', Config.DATA_STRUCTURE[0].label, Config.COMBINED_DICT, f, s)
        assert len(generator.debug_dict['test.nii.gz'].duplicates) == 0

    def test_access_dictionary(self, generator: Generator):
        size = (10, 10, 10)
        f = create_test_image(size)
        s = create_test_image(size)
        result = generator.append_record('test.nii.gz', Config.DATA_STRUCTURE[0].label, Config.COMBINED_DICT, f, s)
        assert generator.access_count_dictionary[Config.DATA_STRUCTURE[0].label] == True

    def test_access_twice_debug_dictionary(self, generator: Generator):
        size = (10, 10, 10)
        f = create_test_image(size)
        s = create_test_image(size)
        result = generator.append_record('test.nii.gz', Config.DATA_STRUCTURE[0].label, Config.COMBINED_DICT, f, s)
        result = generator.append_record('test.nii.gz', Config.DATA_STRUCTURE[0].label, Config.COMBINED_DICT, f, s)
        assert len(generator.debug_dict['test.nii.gz'].duplicates) == 1
