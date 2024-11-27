import os
import cv2
import glob
import json
import re
import shutil
from flask import Flask, request, render_template
from similarity import *

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
SEGMENTED_FOLDER = 'segmented'
OUTPUT_FOLDER = 'static'
COMPARE_FOLDER = 'compare'  # New folder for comparison images

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['COMPARE_FOLDER'] = COMPARE_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(COMPARE_FOLDER, exist_ok=True)


def segment_image(image_path):
    output_json = os.path.splitext(image_path)[0] + '.json'
    kraken_command = f"kraken -i {image_path} {output_json} segment -bl"
    os.system(kraken_command)
    with open(output_json, 'r') as file:
        data = json.load(file)
    lines = data.get('lines', [])
    return lines


def crop_lines(image_path, lines):
    for file_name in os.listdir(OUTPUT_FOLDER):
        if file_name.endswith('.jpg'):
            os.remove(os.path.join(OUTPUT_FOLDER, file_name))

    image = cv2.imread(image_path)
    cropped_images = []
    line_number = 1

    for line in lines:
        if 'boundary' in line:
            boundary = line['boundary']
            x_coords = [point[0] for point in boundary]
            y_coords = [point[1] for point in boundary]
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            if cropped_image.shape[1] < 500 or cropped_image.shape[0] > 300:
                continue

            cropped_image_path = os.path.join(OUTPUT_FOLDER, f'line_{line_number}.jpg')
            cv2.imwrite(cropped_image_path, cropped_image)
            cropped_images.append(cropped_image_path)
            line_number += 1

    return cropped_images


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save uploaded image
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
            file.save(filename)
            
            # Segment the image and crop the lines
            lines = segment_image(filename)
            cropped_images = crop_lines(filename, lines)
        return render_template('segmentation.html', cropped_images=cropped_images)
    return render_template('index.html')


@app.route('/segmentation', methods=['GET'])
def segmentation():
    cropped_images = glob.glob(os.path.join(app.config['OUTPUT_FOLDER'], 'line_*.jpg'))
    cropped_images = [os.path.basename(image) for image in cropped_images]
    cropped_images.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return render_template('segmentation.html', cropped_images=cropped_images)

@app.route('/save_selected_images', methods=['POST'])
def save_selected_images():
    for file_name in os.listdir(OUTPUT_FOLDER):
        if file_name.endswith('.png'):
            os.remove(os.path.join(OUTPUT_FOLDER, file_name))

    selected_lines = request.form.getlist('selected_lines')
    saved_images = []
    for index, image_name in enumerate(selected_lines, start=1):
        source_image_path = os.path.join(app.config['OUTPUT_FOLDER'], image_name)
        destination_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{index}.png')
        shutil.copy(source_image_path, destination_image_path)
        saved_images.append(destination_image_path)
    
    png_files = glob.glob(os.path.join(app.config['OUTPUT_FOLDER'], '*.png'))
    png_files = sorted(png_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    processor = ImageProcessor()
    changed_lines = processor.process_images_list(png_files)
    unique_pairs = list({tuple(sorted(pair)) for pair in changed_lines})
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
    unique_pairs.sort(key=lambda x: extract_number(x[0]))
    print(unique_pairs)
    return render_template('comparison_result.html', lines=unique_pairs)


if __name__ == '__main__':
    app.run(debug=True)
