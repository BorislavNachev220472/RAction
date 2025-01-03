import os
import re
import SimpleITK as sitk

from typing import List, Dict
from dataclasses import dataclass
from specifix.segmentation.cli.config import Config
from specifix.segmentation.cli.converter import Converter
from specifix.segmentation.cli.io import BoneIOFileManager


@dataclass
class DebugEntry:
    """
    Defines the structure of the debug dictionary which in the ends stores some useful information.
    """
    count: int
    duplicates: List[str]
    unknown: List[str]


class Generator(BoneIOFileManager):

    def __init__(self, input_directory: str, output_directory: str, reference_filename: str):
        super().__init__(input_directory, output_directory)
        self.Converter = Converter(reference_filename)
        self.debug_dict: Dict[str, DebugEntry] = {}
        self.access_count_dictionary = {}

    def clear(self):
        """
        This function clears the access_count_dictionary. If some of the lines are checks it can throw errors indicating
        that a single CT scan contains multiple nrrds for a single label.
        :return: None
        """
        self.access_count_dictionary = {}

    def append_record(self, ct_scan: str, filename: str, labels: dict,
                      f: sitk.Image, s: sitk.Image):
        """
        This function appends/stacks one image on top of another image.
        :param ct_scan: str the filename of the nifti file used for debug purposes.
        :param filename: str the filename used for mapping the variation of each label to its original name.
        :param labels: dict which contains all variations of the labels. It should be in the following format.
        key: str, value: object. You can find more information in the Config file.
        :param f: sitk.Image represents the old image that will be used as base. If it's none then the function will
        automatically return the new image as the stacked one.
        :param s: sitk.Image represents the new image that has to be stacked on top of the "prev" image.
        :return: sitk.Image
        """
        if ct_scan not in self.debug_dict.keys():
            self.debug_dict[ct_scan] = DebugEntry(count=0, duplicates=[], unknown=[])

        self.debug_dict[ct_scan].count += 1
        filename = re.sub(Config.NRRD_REGEX_PATTERN, '', os.path.splitext(filename)[0], flags=re.IGNORECASE).strip() \
                   + Config.NRRD_EXTENSION

        if filename in labels:
            if labels[filename].label in self.access_count_dictionary:
                print(
                    f"Duplicate label found for: {filename}. Record {ct_scan} has multiple {labels[filename].label}!")
                self.debug_dict[ct_scan].duplicates.append(filename)
                # raise ValueError(
                #    f"Duplicate label found for: {filename}. Record {input_folder} has multiple {labels[filename]['label']}!"
                # )

            image = self.Converter.stack_nrrd(labels[filename].id, f, s)
            self.access_count_dictionary[labels[filename].label] = True
        else:
            print(f"Unknown label: {filename}. Record {ct_scan} doesn't have a corresponding label variation!")
            # self.debug_dict[ct_scan].unknown.append(filename)
            # image = self.Converter.stack_nrrd(labels[Config.DEFAULT_UNKNOWN].id, f, s)
            image = f

            # raise ValueError(
            #    f"Unknown label: {filename}. Record {input_folder} doesn't have a corresponding label variation!")

        return image

    def process_single_ct_scan(self, ct_scan: str, output_filename: str):
        """
        This function reads all nrrds corresponding to the passed CT scan filename and appends them to a single nifti
        file.
        The CT scan should be stored in the following way:
        CT_SCAN_NAME
            - nrrd
                - *files*
        :param ct_scan: str the name of folder of the CT scan that contains its nrrds.
        :param output_filename: str the output name of the generated nifti file.
        :return: None
        """
        # Create an iterator fromm all files and get the first NRRD file to get image dimensions and spacing
        current_files_iterator = iter((f for f in self
                                      .set_inner_folder(ct_scan)
                                      .set_inner_folder(Config.NRRD_INNER_DIRECTORY)
                                      .read_all() if f.endswith(Config.NRRD_EXTENSION)))
        first_filename = next(current_files_iterator, None)
        if not first_filename:
            raise FileNotFoundError(f"No {Config.NRRD_EXTENSION} files found in directory {ct_scan}.")

        image = self.append_record(ct_scan=ct_scan,
                                   filename=first_filename,
                                   labels=Config.COMBINED_DICT,
                                   f=None,
                                   s=self
                                   .set_inner_folder(ct_scan)
                                   .set_inner_folder(Config.NRRD_INNER_DIRECTORY)
                                   .read_file(first_filename))

        for filename in current_files_iterator:
            image = self.append_record(ct_scan, filename, Config.COMBINED_DICT,
                                       f=image,
                                       s=self
                                       .set_inner_folder(ct_scan)
                                       .set_inner_folder(Config.NRRD_INNER_DIRECTORY)
                                       .read_file(filename))

        # if len(access_count_dictionary.keys()) != total_labels_count:
        #     raise ValueError(
        #         f"Labels count must be {len(labels.keys())}. Record {nrrd_folder} doesn't have all specified labels!"
        #         f"Labels value: {labels.keys()}")
        if image is not None:
            temp = self.Converter.copy_metadata_from_image(image)
            print(temp.GetSpacing())
            self.save_file(image=temp, file_name=os.path.basename(output_filename), should_clean=False)
        else:
            print(f'CT Scan {ct_scan} has no corresponding label variations!')
        self.clear()

    def process_all(self):
        """
        Process multiple CT scans.
        :return: None
        """
        for folder_name in self.read_all():
            output_file_name = f'{folder_name}{Config.NIFTI_EXTENSION}'
            print(folder_name)
            self.process_single_ct_scan(folder_name, output_file_name)
