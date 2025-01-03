import os
import shutil
import SimpleITK as sitk

from abc import ABC, abstractmethod
from specifix.segmentation.cli.io.calback import reset_input_callback, reset_output_callback


class IOManager(ABC):
    """
    Defines the abstract base class for IO managers.
    """

    @abstractmethod
    def read_all(self) -> list[str]:
        pass

    @abstractmethod
    def read_file(self, file_path: str) -> sitk.Image:
        pass

    @abstractmethod
    def save_file(self, image: sitk.Image, file_path: str) -> None:
        pass

    @abstractmethod
    def set_inner_folder(self, inner_dict: str = ''):
        pass

    @abstractmethod
    def set_output_folder(self, output_folder: str = ''):
        pass

    @abstractmethod
    def set_output_relative_to_input_dir(self):
        pass


class BoneIOFileManager(IOManager):

    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.inner_directory = ''
        self.output_folder = ''

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def set_inner_folder(self, inner_directory: str = ''):
        """
        This function utilizes the builder pattern and concatenates the passed directory to the current state of the
        input_directory.
        :param inner_directory: str the current directory hat will be merged with the existing one.
        :return: object
        """
        self.inner_directory = os.path.join(self.inner_directory, inner_directory)
        return self

    def set_output_folder(self, output_folder: str = ''):
        """
        This function utilizes the builder pattern and concatenates the passed directory to the current state of the
        output_directory.
        :param output_folder: str the current directory hat will be merged with the existing one.
        :return: object
        """
        self.output_folder = os.path.join(self.output_folder, output_folder)
        return self

    def set_output_relative_to_input_dir(self):
        """
        This function utilizes the builder pattern and sets the output directory to be relative to the input directory.
        :return: object
        """
        self.output_directory = os.path.join(self.input_directory, self.inner_directory)
        return self

    @reset_input_callback
    @reset_output_callback
    def build(self):
        """
        This function `builds` the object as it permanently sets the input directory to become the old onput directory
        with the current state of the inner_directory. The same goes for the output directory.
        :return: object
        """
        self.output_directory = os.path.join(self.output_directory, self.output_folder)
        self.input_directory = os.path.join(self.input_directory, self.inner_directory)
        return self

    @reset_input_callback
    def read_all(self) -> list[str]:
        """
        Reads all files at the current input directory. The directory is relative to the input directory as the
        specified current inner directory is merged.
        :return: list[str] with the names of all files in the that directory.
        """
        return os.listdir(os.path.join(self.input_directory, self.inner_directory))

    @reset_input_callback
    def read_file(self, file_name: str) -> sitk.Image:
        """
        Reads a file at a specific location. The location is relative to the input directory as the specified current
        inner directory is merged.
        :param file_name: str the input name ofthe file that should be read.
        :return: sitl.Image
        """
        absolute_path = os.path.join(self.input_directory, self.inner_directory, file_name)
        print(f"Reading file: '{absolute_path}'.")
        return sitk.ReadImage(absolute_path)

    @reset_output_callback
    def save_file(self, image: sitk.Image, file_name: str, should_clean=True) -> None:
        """
        Writes a file at a specific location in a compressed way. The location is relative to the output directory as
        the specified current output directory is merged. It can clear the folder if required.
        :param image: sitk.Image that should be saved.
        :param file_name: str the output name of the desired file
        :param should_clean: bool specifies if the folder should be cleared or not.
        :return: None
        """
        current_directory = os.path.join(self.output_directory, self.output_folder)
        if os.path.exists(current_directory) and should_clean:
            shutil.rmtree(current_directory)

        if not os.path.exists(current_directory):
            os.makedirs(current_directory)

        absolute_path = os.path.join(current_directory, file_name)
        sitk.WriteImage(image, absolute_path, useCompression=True)
        print(f"Saved file: '{absolute_path}'.")
