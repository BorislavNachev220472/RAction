import os
import numpy as np
import SimpleITK as sitk
# from warnings import deprecated

from specifix.segmentation.cli.config import Config
from specifix.segmentation.cli.io import BoneIOFileManager


class Converter(object):

    def __init__(self, reference_filename: str):
        self.reference_image = sitk.ReadImage(reference_filename)

    # @deprecated
    def preprocess_image(self, image: sitk.Image) -> sitk.Image:
        """
        DEPRECATED The idea of this function was to align the misaligned nifti files.
        :param image:
        :return:
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.reference_image)
        # resampler.SetOutputDirection(self.reference_image.GetDirection())
        # resampler.SetOutputOrigin(self.reference_image.GetOrigin())
        # resampler.SetOutputSpacing(self.reference_image.GetSpacing())
        return resampler.Execute(image)

    def stack_nrrd(self, label: str, prev: sitk.Image, new: sitk.Image):
        """
        This method stacks two images on top of each other. They should have teh same dimensions and channels. It's
        implemented in a way that it can be called recursively multiple times. To start the loop it should first be
        called with 'None' as parameter for the 'prev' argument as there won't abe a 'prev' argument during this time.

        :param label: the string representation of the label.
        :param prev: sitk.Image represents the old image that will be used as base. If it's none then the function will
        automatically return the new image as the stacked one.
        :param new: sitk.Image represents the new image that has to be stacked on top of the "prev" image.
        :return: sitk.Image
        """
        if label is not None:
            new = sitk.Cast(new, sitk.sitkInt8)
            new[new > 0] = label

        if prev is None:
            return new

        if prev.GetSize() != new.GetSize():
            new = sitk.Resample(new, prev)

        return sitk.Maximum(prev, new)

    def copy_metadata_from_image(self, image: sitk.Image, label=None):
        """
        This function copies the metadata from an sitk.Image and optionally filters it for a specific label.
        :param image: sitk.Image The image that contains the desired metadata.
        :param label:  int Specifies if the sitk.Image should be filtered for a specific label.
        :return: sitk.Image
        """
        data = sitk.GetArrayFromImage(image)
        return self.copy_metadata_from_array(image, data, label)

    def copy_metadata_from_array(self, image: sitk.Image, array: np.array, label=None):
        """
        This function copies the metadata from an image to a new image which is generated from an numpy array and
        filtered for a specific label. If the label is `None`, then the metadata is copied to the original np.ndarray.
        :param image: sitk.Image The image that contains the desired metadata.
        :param array: np.ndarray The numpy array that should receive the metadata after it's converted to sitk.Image.
        :param label: int Specifies if the array should be filtered for a specific label.
        :return: sitk.Image
        """
        if label is not None:
            array = (array == label).astype(np.uint8)

        image_sitk = sitk.GetImageFromArray(array)
        image_sitk.CopyInformation(image)
        return image_sitk

    def get_array_from_image(self, image: sitk.Image) -> np.ndarray:
        """
        This method abstracts the original GetArrayFromImage function. It's made in this way because later on the
        library used to manipulate the CT scans can be easily changed.
        :param image: sitk.Image
        :return: np.ndarray.
        """
        return sitk.GetArrayFromImage(image)

    def get_image_from_array(self, array: np.array) -> sitk.Image:
        """
            This method abstracts the original GetImageFromArray function. It's made in this way because later on the
            library used to manipulate the CT scans can be easily changed.
            :param image: np.ndarray
            :return: sitk.Imag
        """
        return sitk.GetImageFromArray(array)


class Nifti(BoneIOFileManager):

    def __init__(self, input_directory: str, output_folder: str, reference_filename: str):
        super().__init__(input_directory, output_folder)
        self.Converter = Converter(reference_filename)

    def to_label_tiffs(self, nifti_filename: str) -> list[float]:
        """
        Converts one nifti file to tiffs stored in a separate folder with the following structure:
        unique label 1
            - *tiffs*
        unique label 2
            - *tiffs*
        unique label 3
            - *tiffs*
        for each unique label file.
        :param nifti_filename: str representing the nifti filename that should be converted.
        :return: int returns the spacing of the nifti file."""
        image = self.read_file(nifti_filename)

        label_array = sitk.GetArrayFromImage(image)
        unique_labels = np.unique(label_array)
        unique_labels = unique_labels[unique_labels != 0]
        print(f"Unique labels found: {unique_labels}")
        for label in unique_labels:
            # filename = os.path.basename(nifti_filename).replace(Config.NIFTI_EXTENSION, '')
            current_image = self.Converter.copy_metadata_from_array(image, label_array, label=label)

            for i in range(current_image.GetSize()[2]):
                image_slice = current_image[:, :, i]
                (self.set_inner_folder(Config.DATA_STRUCTURE[label - 1].label).save_file(
                    image=image_slice,
                    file_name=f'{str(i + 1).zfill(5)}{Config.TIFF_EXTENSION}',
                    should_clean=False))
        return image.GetSpacing()

    def to_single_label_tiffs(self, nifti_filename: str) -> list[float]:
        """
        Converts multiple labels to a single tiff folder. Can be used to merge multiple labels into one (binary)
        if required.
        Config.COMBINED_LABEL
            - *tiffs*
        :param nifti_filename: str representing the nifti filename that should be converted.
        :return: int returns the spacing of the nifti file.
        """
        image = self.read_file(nifti_filename)

        label_array = sitk.GetArrayFromImage(image)
        unique_labels = np.unique(label_array)
        unique_labels = unique_labels[unique_labels != 0]
        print(f"Unique labels found: {unique_labels}")
        stacked_image = None
        for label in unique_labels:
            current_image = self.Converter.copy_metadata_from_array(image, label_array, label=label)
            stacked_image = self.Converter.stack_nrrd(None, stacked_image, current_image)

        for i in range(image.GetSize()[2]):
            image_slice = stacked_image[:, :, i]
            (self.set_output_folder(Config.COMBINED_LABEL).save_file(
                image=image_slice,
                file_name=f'{str(i + 1).zfill(5)}{Config.TIFF_EXTENSION}',
                should_clean=False))

        return image.GetSpacing()

    def to_label_nrrds(self, nifti_filename: str) -> list[float]:
        """
        Converts one nifti file to multiple nrrds as it created a folder structure of:
        CT_NAME
            - nrrd
                - *converted files*
        for each nifti file.
        :param nifti_filename: str representing the nifti filename that should be converted.
        :return: int returns the spacing of the nifti file.
        """
        image = self.read_file(nifti_filename)

        label_array = sitk.GetArrayFromImage(image)
        unique_labels = np.unique(label_array)
        unique_labels = unique_labels[unique_labels != 0]
        print(f"Unique labels found: {unique_labels}")
        for label in unique_labels:
            filename = os.path.basename(nifti_filename).replace(Config.NIFTI_EXTENSION, '')

            (self.set_output_folder(filename).set_output_folder(Config.NRRD_INNER_DIRECTORY).save_file(
                image=self.Converter.copy_metadata_from_array(image, label_array, label=label),
                file_name=f'{Config.DATA_STRUCTURE[label - 1].label}{Config.NRRD_EXTENSION}',
                should_clean=False))
        return image.GetSpacing()

    # @deprecated
    def align(self, nifti_filename: str) -> None:
        """
        DEPRECATED The idea of this function was to align the misaligned nifti files.
        :param nifti_filename:
        :return:
        """
        image = self.read_file(nifti_filename)

        image_array = sitk.GetArrayFromImage(image)
        new_order = (2, 1, 0)
        reordered_array = np.transpose(image_array, new_order)

        reordered_array = reordered_array[:, ::-1, :]

        image = sitk.GetImageFromArray(reordered_array)

        image.SetDirection(self.Converter.reference_image.GetDirection())
        image.SetOrigin(self.Converter.reference_image.GetSpacing())
        image.SetSpacing(self.Converter.reference_image.GetOrigin())

        self.save_file(image=self.Converter.copy_metadata_from_image(image),
                       file_name=os.path.basename(nifti_filename))

    # @deprecated
    def align_all(self):
        """
        DEPRECATED
        :return:
        """
        for filename in os.listdir(self.input_directory):
            self.align(filename)

    def convert_all(self):
        """
            Converts multiple nifti files to nrrds. Each nifti file will be stored in a separate folder as explained in
            the self.to_label_nrrds function.
        """
        for filename in os.listdir(self.input_directory):
            self.to_label_nrrds(filename)
