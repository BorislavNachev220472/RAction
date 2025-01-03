import re
import os
from dataclasses import dataclass
from typing import List, Dict

import SimpleITK as sitk
import nibabel as nib
import numpy as np


@dataclass
class Label(object):
    id: int
    label: str
    variations: List[str]

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        return {variation: {"label": self.label, "id": self.id} for variation in self.variations}


debug_dict = {}


# Function to load an NRRD file and return the SimpleITK Image object
def load_nrrd_file(file_path):
    return sitk.ReadImage(file_path)


# Function to process a 2D slice from a 3D fragment
def process_slice(slice_array, label_value):
    processed_slice = np.where(slice_array == 1, label_value, 0)
    return processed_slice


# Function to convert a SimpleITK image to a NIfTI file and save it
def convert_to_nifti(sitk_image, output_file_path):
    array = sitk.GetArrayFromImage(sitk_image)
    spacing = np.array(sitk_image.GetSpacing())
    origin = np.array(sitk_image.GetOrigin())
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)

    affine = np.eye(4)
    affine[:3, :3] = direction * spacing[:, np.newaxis]
    affine[:3, 3] = origin

    nifti_image = nib.Nifti1Image(array, affine)
    nib.save(nifti_image, output_file_path)
    # print(f"NIfTI file saved at: {output_file_path}")


def process_nrrd(current_folder: str, filename: str, label_value: int, final_array: []) -> []:
    file_path = os.path.join(current_folder, filename)
    itk_fragment = sitk.ReadImage(file_path)
    fragment_array = sitk.GetArrayFromImage(itk_fragment)

    for i in range(fragment_array.shape[0]):
        processed_slice = process_slice(fragment_array[i, :, :], label_value)
        final_array[i, :, :] = np.where(final_array[i, :, :] == 0, processed_slice, final_array[i, :, :])

    return final_array


def append_record(access_count_dictionary, input_folder: str, filename: str, labels: dict, final_array: []):
    ct_scan = input_folder.split('/')[-2]
    if ct_scan not in debug_dict:
        debug_dict[ct_scan] = {'count': 0, 'duplicates':[], 'unknown': []}
            
    debug_dict[ct_scan]['count'] += 1
    filename_processed = re.sub(r'\b(fin|final|-)\b', '', os.path.splitext(filename)[0], flags=re.IGNORECASE).strip() + '.nrrd'
    
    if filename_processed in labels:
        if labels[filename_processed]['label'] in access_count_dictionary:
            print(f"Duplicate label found for: {filename}. Record {input_folder} has multiple {labels[filename_processed]['label']}!")
            debug_dict[ct_scan]['duplicates'].append(filename)
            #raise ValueError(
            #    f"Duplicate label found for: {filename}. Record {input_folder} has multiple {labels[filename]['label']}!"
            #)
        
        final_array = process_nrrd(input_folder, filename, labels[filename_processed]['id'], final_array)
        access_count_dictionary[labels[filename_processed]['label']] = True
    else:
        print(f"Unknown label: {filename}. Record {input_folder} doesn't have a corresponding label variation!")
        debug_dict[ct_scan]['unknown'].append(filename)
        # final_array = process_nrrd(input_folder, filename, label_value=labels['unknown']['id'], final_array=final_array)
        # raise ValueError(
        #    f"Unknown label: {filename}. Record {input_folder} doesn't have a corresponding label variation!")

    return final_array


# Function to merge segmentation fragments from .nrrd files and save as a single .nii.gz file
def merge_fragments_and_save_nifti(access_count_dictionary, input_folder: str, labels: dict, nrrd_folder: str, output_file_path: str):
    # Create an iterator fromm all files and get the first NRRD file to get image dimensions and spacing
    current_files_iterator = iter((f for f in os.listdir(nrrd_folder) if f.endswith('.nrrd')))
    first_filename = next(current_files_iterator, None)
    if not first_filename:
        raise FileNotFoundError(f"No .nrrd files found in directory {nrrd_folder}.")

    itk_image = sitk.ReadImage(os.path.join(input_folder, first_filename))
    first_array = sitk.GetArrayFromImage(itk_image)
    final_array = append_record(access_count_dictionary=access_count_dictionary,
                                input_folder=input_folder,
                                filename=first_filename,
                                labels=labels,
                                final_array=np.zeros_like(first_array, dtype=np.uint8))

    for filename in current_files_iterator:
        final_array = append_record(access_count_dictionary, input_folder, filename, labels, final_array)

    # if len(access_count_dictionary.keys()) != total_labels_count:
    #     raise ValueError(
    #         f"Labels count must be {len(labels.keys())}. Record {nrrd_folder} doesn't have all specified labels!"
    #         f"Labels value: {labels.keys()}")

    # Create a SimpleITK image from the final array
    final_itk_image = sitk.GetImageFromArray(final_array)
    final_itk_image.SetSpacing(itk_image.GetSpacing())
    final_itk_image.SetOrigin(itk_image.GetOrigin())
    final_itk_image.SetDirection(itk_image.GetDirection())

    # Save as NIfTI
    convert_to_nifti(final_itk_image, output_file_path)


# Main function to process a single folder
def process_single_folder(input_folder: str, labels: dict, nrrd_folder: str, output_folder: str,
                          output_file_name: str):
    
    access_count_dictionary = {}
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file_path = os.path.join(output_folder, f"{output_file_name}.nii.gz")
    merge_fragments_and_save_nifti(access_count_dictionary, input_folder, labels, nrrd_folder, output_file_path)


# Example usage for a single folder
if __name__ == "__main__":
    output_folder = r"E:\Projects\Buas\Year3\2024-25ab-fai3-specialisation-project-team-specifix-1\data\Masks_&_Meshes\F034\output2"
    output_file_name = "merged_labels4"

    data_structure = [
        Label(id=1, label="radial", variations=['R.nrrd']),
        Label(id=2, label="ulnar", variations=['pU.nrrd']),
        Label(id=3, label="metaphysis", variations=['M1.nrrd', 'M0.nrrd']),
        Label(id=4, label="dR", variations=['dR.nrrd', 'dDR.nrrd', 'dVR.nrrd', 'dRU.nrrd']),
        Label(id=5, label="dD", variations=['dD.nrrd', 'dDC.nrrd', 'dDRU.nrrd']),
        Label(id=6, label="dV", variations=['dV.nrrd', 'dVC.nrrd', 'dVRU.nrrd']),
        Label(id=7, label="dU", variations=['dU.nrrd', 'dDU.nrrd', 'dDU1.nrrd', 'dDU2.nrrd', 'dVU.nrrd', 'dLU.nrrd']),
        Label(id=8, label="dC", variations=['dC.nrrd', 'dL.nrrd']),
        Label(id=9, label="dorso-ulnar",
              variations=['p1DU1', 'p1DU', 'pDU', 'pDU1', 'DU', 'p1DU1', 'p1DU2', 'pDU2', 'p1DU3', 'pDU3', 'p2DU1',
                          'p2DU', 'p2DU1', 'p2DU2', 'p2DU3']),
        Label(id=10, label="dorso-radial",
              variations=['p1DR1', 'pDR', 'p1DR', 'DR', 'pDR1', 'p1DR1', 'p1DR2', 'pDR2', 'p1DR3', 'pDR3', 'L', 'p1L',
                          'p2L', 'pDC', 'p2DR', 'p2DR1', 'p2DR2', 'p2DR3']),
        Label(id=11, label="volar-radial",
              variations=['p1VR1', 'pVR', 'p1VR1', 'pVR1', 'p1VR', 'p1VR2', 'pVR2', 'p1VR2', 'VR', 'p2VR1', 'p2VR',
                          'p2VR1', 'p2VR2', 'pV']),
        Label(id=12, label="volar-ulnar",
              variations=['p1VU', 'p1VU1', 'VU', 'pVU', 'pVU1', 'p1VU1', 'p1VU', 'p1VU2', 'p1VU2', 'pVU2', 'VU',
                          'p2VU1', 'p2VU', 'p2VU1', 'p2VU2']),
        Label(id=13, label="ulna",
              variations=['Ulna broken.nrrd', 'Ulna Broken.nrrd', 'Ulna CL.nrrd', 'Ulna.nrrd', 'Ulna Borken.nrrd']),
        Label(id=14, label="dINTACT", variations=['dINTACT.nrrd']),
        Label(id=15, label="unknown", variations=['unknown'])
    ]

    combined_dict = {}

    for label in data_structure:
        combined_dict.update(label.to_dict())

    # Process the folder and generate the NIfTI file
    process_single_folder(combined_dict, nrrd_folder, output_folder, output_file_name)
    print(debug_dict)
