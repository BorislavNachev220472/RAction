import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib


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
    print(f"NIfTI file saved at: {output_file_path}")


def process_nrrd(current_folder: str, filename: str, label_value: int, final_array: []) -> []:
    file_path = os.path.join(current_folder, filename)
    itk_fragment = sitk.ReadImage(file_path)
    fragment_array = sitk.GetArrayFromImage(itk_fragment)

    for i in range(fragment_array.shape[0]):
        processed_slice = process_slice(fragment_array[i, :, :], label_value)
        final_array[i, :, :] = np.where(final_array[i, :, :] == 0, processed_slice, final_array[i, :, :])

    return final_array


# Function to merge segmentation fragments from .nrrd files and save as a single .nii.gz file
def merge_fragments_and_save_nifti(nrrd_folder, output_file_path):
    # Find the first NRRD file to get image dimensions and spacing
    first_file = next((f for f in os.listdir(nrrd_folder) if f.endswith('.nrrd')), None)
    if not first_file:
        print("No .nrrd files found.")
        return

    itk_image = sitk.ReadImage(os.path.join(nrrd_folder, first_file))
    first_array = sitk.GetArrayFromImage(itk_image)
    final_array = np.zeros_like(first_array, dtype=np.uint8)

    label_value = 1  # Start label numbering from 1
    ulna_variations = ['ulna.nrrd', 'ulna broken.nrrd', 'borken.nrrd']
    m1_variations = ['m1.nrrd', 'm0.nrrd']

    current_folder_files_lowered = np.array(list(map(lambda x: x.lower(), os.listdir(nrrd_folder))))
    current_folder_files_ = os.listdir(nrrd_folder)

    for filename in m1_variations:
        idx = np.where(current_folder_files_lowered == filename)[0]
        if  len(idx) > 0:
            final_array = process_nrrd(nrrd_folder, current_folder_files_[idx[0]], label_value, final_array)
    label_value += 1

    for filename in ulna_variations:
        idx = np.where(current_folder_files_lowered == filename)[0]
        if len(idx) > 0:
            final_array = process_nrrd(nrrd_folder, current_folder_files_[idx[0]], label_value, final_array)
    label_value += 1

    if label_value != 3:
        raise IndexError(f"Label value must be 3. Record {nrrd_folder} is without both M1 and Ulna!")

    for idx, filename in enumerate(list(current_folder_files_lowered)):
        if filename.endswith('.nrrd') \
                and not any(
            sub in filename.lower() for sub in ["region", "small", "yellow", "cyan", "green", "radius"]) \
                and filename not in m1_variations and filename not in ulna_variations:
            final_array = process_nrrd(nrrd_folder, current_folder_files_[idx], label_value, final_array)

    # Create a SimpleITK image from the final array
    final_itk_image = sitk.GetImageFromArray(final_array)
    final_itk_image.SetSpacing(itk_image.GetSpacing())
    final_itk_image.SetOrigin(itk_image.GetOrigin())
    final_itk_image.SetDirection(itk_image.GetDirection())

    # Save as NIfTI
    convert_to_nifti(final_itk_image, output_file_path)


# Main function to process a single folder
def process_single_folder(nrrd_folder, output_folder, output_file_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file_path = os.path.join(output_folder, f"{output_file_name}.nii.gz")
    merge_fragments_and_save_nifti(nrrd_folder, output_file_path)


# Example usage for a single folder
if __name__ == "__main__":
    nrrd_folder = r"D:\Masks\F020\Radius and ulna\Masks\Segmentation"
    output_folder = r"C:\Users\soham\OneDrive\Desktop\tmp\output_single"
    output_file_name = "merged_labels"

    # Process the folder and generate the NIfTI file
    process_single_folder(nrrd_folder, output_folder, output_file_name)
