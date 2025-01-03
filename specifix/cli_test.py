if __name__ == '__main__':
    # %%----------------------------------
    from specifix.segmentation.cli.shell import Shell

    reference_filename = '../../data/Images_TR/F020_0000.nii.gz'
    input_directory = '../../data/Masks_&_Meshes'
    output_directory = '../../data/soha_nrrd/shelled'
    shell = Shell(input_directory, output_directory, reference_filename)

    shell.process_ct_scan(ct_scan='F020', labels_output_path='shell.nii.gz')

    # %% ----------------------------------
    from specifix.segmentation.cli.generator import Generator

    reference_filename = '../../data/Images_TR/F020_0000.nii.gz'
    input_directory = '../../data/Masks_&_Meshes'
    output_directory = '../../data/soha_nrrd/generated'
    shell = Generator(input_directory, output_directory, reference_filename)

    shell.process_single_ct_scan(ct_scan='F020', output_filename='generated.nii.gz')

    # %% ----------------------------------
    from specifix.segmentation.cli.converter import Nifti

    reference_filename = '../../data/Images_TR/F020_0000.nii.gz'
    input_directory = '../../data/Masks_&_Meshes'
    output_directory = '../../data/soha_nrrd/converted'
    converter = Nifti(input_directory, output_directory, reference_filename)

    converter.to_label_nrrds(nifti_filename='F020')

    # %% ----------------------------------
    from specifix.segmentation.cli.mesh import Mesh

    reference_filename = '../../data/Images_TR/F020_0000.nii.gz'
    input_directory = '../../data/Masks_&_Meshes'
    output_directory = '../../data/soha_nrrd/meshed'
    shell = Mesh(input_directory, output_directory, reference_filename)

    shell.generate_mesh(nifti_filename='F020')
