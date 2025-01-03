import os
import subprocess
from dotenv import load_dotenv
from flask import Flask, request, render_template, send_from_directory, jsonify

app = Flask(__name__)

# Directory to store uploaded files
UPLOAD_FOLDER = '/workspace/uploads/'
DOWNLOAD_FOLDER = '/workspace/downloads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)
    
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER


@app.route("/")
def homepage():
    """
    Homepage to list uploaded files with download links.
    """
    files = [el for el in os.listdir(app.config["DOWNLOAD_FOLDER"]) if '.nii.gz' in el] 
    return render_template("homepage.html", files=files)

@app.route("/upload_files")
def uf():
    """
    Homepage to list uploaded files with download links.
    """
    files = [el for el in os.listdir(app.config["UPLOAD_FOLDER"]) if '.nii.gz' in el] 
    return files

@app.route("/download_files")
def df():
    """
    Homepage to list uploaded files with download links.
    """
    files = [el for el in os.listdir(app.config["DOWNLOAD_FOLDER"]) if '.nii.gz' in el] 
    return files

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    API endpoint to upload a file.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    return jsonify({"message": f"File '{file.filename}' uploaded successfully!"}), 200


def set_envs():
    data = {
        'nUNet_raw':'/workspace/data/segmentation/inference/nnUNet_raw',
        'nnUNet_preprocessed':'/workspace/data/segmentation/inference/nnUNet_preprocessed',
        'nnUNet_results':'/workspace/data/segmentation/inference/nnUNet_results'
    }
    # data = {
    # 'nnUNet_raw':'/workspace/2024-25ab-fai3-specialisation-project-team-specifix-1/data/segmentation/nnUNet_raw',
    # 'nnUNet_preprocessed':'/workspace/2024-25ab-fai3-specialisation-project-team-specifix-1/data/segmentation/nnUNet_preprocessed',
    # 'nnUNet_results':'/workspace/2024-25ab-fai3-specialisation-project-team-specifix-1/data/segmentation/nnUNet_results'
    # }


    with open('.env', 'w') as file:
        for key in data:
            file.write(f'{key}={data[key]}\n')

    with open('.env', 'r') as file:
        cf = file.read()
    print(cf)
    


@app.route("/detect/<modelname>", methods=["GET"])
def detect_file(modelname):
    """
    API endpoint to getect bones.
    """
    set_envs()
    # Load environment variables from .env file
    load_dotenv(override=True)
    output_file = os.path.join(app.config["UPLOAD_FOLDER"], 'output.nii.gz')
    command = [
        "nnUNetv2_predict",
        "-i", f"{app.config['UPLOAD_FOLDER']}",
        "-o", f"{app.config['DOWNLOAD_FOLDER']}",
        "-d", f"{modelname}",
        "-c", "3d_fullres",
        "-f", "0"
    ]
    subprocess.run(command, check=True)
    
    return jsonify({"message": f"Detection saved as '{output_file}' successfully!",
                    "filename": output_file}), 200


@app.route("/download/<filename>")
def download_file(filename):
    """
    API endpoint to download a file.
    """
    return send_from_directory(app.config["DOWNLOAD_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
