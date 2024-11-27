import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from similarity import *
from pre_processor import *
from feature_extractor import *
from line_segmentor import *
import glob
import os


class ImageProcessor:
    def __init__(self, detection_method='FAST', top_n=500):
        self.detection_method = detection_method
        self.top_n = top_n

    def improve_image(self, image):
        """Enhance the image by applying Gaussian Blur and subtracting from original."""
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        subtracted = cv2.subtract(image, blurred)
        enhanced = cv2.addWeighted(image, 0.75, subtracted, 0.25, 0)
        return enhanced

    def get_cropped(self, image1, image2, image3):
        """Crop images to the smallest size among them."""
        min_height = min(image1.shape[0], image2.shape[0], image3.shape[0])
        min_width = min(image1.shape[1], image2.shape[1], image3.shape[1])

        def crop_center(image, target_height, target_width):
            height, width = image.shape[:2]
            top = max((height - target_height) // 2, 0)
            left = max((width - target_width) // 2, 0)
            return image[top:top + target_height, left:left + target_width]

        return (crop_center(image1, min_height, min_width),
                crop_center(image2, min_height, min_width),
                crop_center(image3, min_height, min_width))

    def process_images(self, img):
        """Extract keypoints and descriptors from an image."""
        if self.detection_method == 'SIFT':
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(img, None)
        elif self.detection_method == 'FAST':
            fast = cv2.FastFeatureDetector_create()
            kp = fast.detect(img, None)
            kp = sorted(kp, key=lambda x: x.response, reverse=True)[:self.top_n]
            sift = cv2.SIFT_create()
            kp, des = sift.compute(img, kp)
        else:
            raise ValueError("Detection method must be 'SIFT' or 'FAST'")
        return kp, des
    
    def classify_and_evaluate(self, des1, des2, des3):
        """Classify descriptors of the third image and evaluate similarity."""
        labels1 = np.zeros(len(des1))
        labels2 = np.ones(len(des2))
        X_train = np.vstack((des1, des2))
        y_train = np.hstack((labels1, labels2))

        classifier = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11))
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(des3)
        similarity_to_image1 = np.mean(y_pred == 0) * 100
        similarity_to_image2 = np.mean(y_pred == 1) * 100

        return similarity_to_image1, similarity_to_image2, y_pred

    def get_lbp(self, images):
        pre_processed = PreProcessor()
        results = []
    
        if len(images) != 3:
            raise ValueError("You must pass exactly 3 images.")
    
        processed_images = []
        for image in images:
            gray_image, binary_image = pre_processed.process(image)
            processed_images.append((gray_image, binary_image))
     
        segmented_images = []
        for gray_image, binary_image in processed_images:
            segmentor = LineSegmentor(gray_image, binary_image)
            gray_lines, bin_lines = segmentor.segment()
            segmented_images.append((gray_lines, bin_lines))
    
        feature_vectors = []
        for gray_lines, bin_lines in segmented_images:
            feature_extractor = FeatureExtractor(gray_lines, bin_lines)
            features = feature_extractor.extract()
            feature_vectors.append(np.array(features))
        
        for i in range(len(feature_vectors)):
            for j in range(i + 1, len(feature_vectors)):
                diff_score = int(np.sum(np.abs(feature_vectors[i] - feature_vectors[j])))
                results.append(diff_score)

        return np.mean(results)

    def process_images_list(self, image_paths):
        lines = []
        for i in range(len(image_paths) - 2):
            # Load images in grayscale
            image1 = cv2.imread(image_paths[i], 0)
            image2 = cv2.imread(image_paths[i + 2], 0)
            image3 = cv2.imread(image_paths[i + 1], 0)

            # Crop images
            image1, image2, image3 = self.get_cropped(image1, image2, image3)
            image1 = self.improve_image(image1)
            image2 = self.improve_image(image2)
            image3 = self.improve_image(image3)

            # Process images to get keypoints and descriptors
            _, descriptors1 = self.process_images(image1)
            _, descriptors2 = self.process_images(image2)
            _, descriptors3 = self.process_images(image3)

            similarity1, similarity2, predictions = self.classify_and_evaluate(descriptors1, descriptors2, descriptors3)
            results = self.get_lbp([image1, image2, image3])
            print(int(results))
            if abs(max(similarity1, similarity2) - min(similarity1, similarity2)) <= 50:
                if results <= 70:
                    print('same')
                elif min(similarity1, similarity2) == similarity1:
                    lines.append((image_paths[i], image_paths[i+1]))
                    print(f'{image_paths[i]} has changed w.r.t {image_paths[i+1]}')
                else:
                    lines.append((image_paths[i + 2], image_paths[i+1]))
                    print(f'{image_paths[i + 2]} has changed w.r.t {image_paths[i+1]}')

            elif abs(max(similarity1, similarity2) - min(similarity1, similarity2)) > 50 and results >= 50:
                if min(similarity1, similarity2) == similarity1:
                    lines.append((image_paths[i], image_paths[i+1]))
                    print(f'{image_paths[i]} has changed w.r.t {image_paths[i+1]}')
                else:
                    lines.append((image_paths[i + 2], image_paths[i+1]))
                    print(f'{image_paths[i + 2]} has changed w.r.t {image_paths[i+1]}')
            else:
                print('same')

            print(image_paths[i], image_paths[i + 2], image_paths[i + 1])
            print(f"Similarity of Image 3 to Image 1: {similarity1:.2f}%")
            print(f"Similarity of Image 3 to Image 2: {similarity2:.2f}%")
            print('----------------------------------------------------------')

        return lines


# Usage example:
if __name__ == "__main__":
    # The path to PNG files would be dynamically determined.
    png_files = glob.glob(os.path.join('', '*.png'))
    png_files = sorted([f for f in png_files if f.split('.')[0].isdigit()], key=lambda x: int(x.split('.')[0]))

    processor = ImageProcessor()
    changed_lines = processor.process_images_list(png_files)
    print("Lines with changes:", changed_lines)