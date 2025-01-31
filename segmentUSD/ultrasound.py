import cv2
import os
import pydicom
import numpy as np
import nrrd


class UltrasoundSegmentation():
    """
    :param
        input_path - input path to your dicom image
        output_path - output path where new image and segmentation mask would be saved
        pixel_number - count of header pixel
    """
    def __init__(self, input_path, pixel_number = 70, output_path = None):
        self.input_path = input_path
        self.output_path = output_path
        self.pixel_number = pixel_number


    def dicom2nrrdImage(self):
        _, name = os.path.split(self.input_path)
        dicom_data = pydicom.dcmread(self.input_path)
        image_array = dicom_data.pixel_array
        cropped_image = image_array[self.pixel_number:, :]
        image_array = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image[..., np.newaxis]
        gray_image = cv2.flip(gray_image, 0)
        if self.output_path is not None:
            save_path = os.path.join(self.output_path, f"{name}_new_image.nrrd")
            nrrd.write(save_path, gray_image)
        else:
            nrrd.write(f"{name}_new_image.nrrd", gray_image)


    def dicom2nrrdMask(self):
        self.pros_dicom(self.input_path, self.pixel_number, self.output_path)

    def connect_white_pixels(self, binary_image, max_gap=30):
        height, width = binary_image.shape
        result_image = binary_image.copy()
        for y in range(height):
            white_pixels = np.where(binary_image[y] == 255)[0]
            if len(white_pixels) > 0:
                start_idx = white_pixels[0]
                last_idx = white_pixels[0]
                for i in range(1, len(white_pixels)):
                    gap = white_pixels[i] - white_pixels[i - 1]
                    if gap <= max_gap:
                        last_idx = white_pixels[i]
                    else:
                        result_image[y, start_idx:last_idx + 1] = 255
                        start_idx = white_pixels[i]
                        last_idx = white_pixels[i]
                result_image[y, start_idx:last_idx + 1] = 255
        return result_image

    def color_black_between_whites_1(self, binary_image):
        height, width = binary_image.shape
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            found_first_white = False
            black_count = 0
            for y in range(height):
                if binary_image[y, x] == 255:
                    if found_first_white:
                        for j in range(y - black_count, y):
                            colored_image[j, x] = [0, 0, 255]
                    found_first_white = True
                    black_count = 0
                elif found_first_white and binary_image[y, x] == 0:
                    black_count += 1
        white_colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        white_colored_image[binary_image == 255] = [0, 0, 255]
        result_image = np.maximum(colored_image, white_colored_image)
        return result_image

    def remove_black_inside_red(self, colored_image, n=0):
        height, width, _ = colored_image.shape
        red_color = [0, 0, 255]
        for x in range(width):
            for y in range(height):
                if np.array_equal(colored_image[y, x], red_color):
                    for dx in range(-n, n + 1):
                        for dy in range(-n, n + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if np.array_equal(colored_image[ny, nx], [0, 0, 0]):
                                    colored_image[y, x] = [0, 0, 0]
        return colored_image


    def pros_dicom(self, input_path, pixel_number, output_path):
        #extarct file name
        _, name = os.path.split(input_path)
        dicom_data = pydicom.dcmread(input_path)
        #get pixel data from dicom file
        image_array = dicom_data.pixel_array

        cropped_image = image_array[pixel_number:, :]
        image_array = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_array = cv2.flip(image_array, 0)
        blue_channel = image_array[:, :, 0]
        _, thresh = cv2.threshold(blue_channel, 254, 255, cv2.THRESH_BINARY)

        # find all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a contour around each rectangle that was found in the original image
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
        B = image_array[:, :, 0]
        G = image_array[:, :, 1]
        R = image_array[:, :, 2]

        yellow_mask = (R >= 254) & (R <= 255) & (G >= 254) & (G <= 255) & (B >= 0) & (B <= 0)
        yellow_only_mask = np.zeros_like(image_array)
        yellow_only_mask[yellow_mask] = [0, 255, 255]  # Сохраним маску только жёлтой области
        yellow_gray = cv2.cvtColor(yellow_only_mask, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # Эллиптическое ядро 7x7
        closed_yellow_area = cv2.morphologyEx(yellow_gray, cv2.MORPH_CLOSE, kernel)
        _, binary_image = cv2.threshold(closed_yellow_area, 200, 255, cv2.THRESH_BINARY)
        connected_image = self.connect_white_pixels(binary_image)
        image = self.color_black_between_whites_1(connected_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            mask = np.zeros_like(image)
            n = 1
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    for point in contour:
                        x, y = point[0]
                        nx, ny = x + i, y + j
                        if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                            image[ny, nx] = 0
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        if len(contours) > 1:
            print(f"Found {len(contours)} figures.")
            areas = [cv2.contourArea(c) for c in contours]
            min_area_idx = np.argmin(areas)
            cv2.drawContours(image, [contours[min_area_idx]], -1, (0, 0, 0), thickness=cv2.FILLED)
        else:
            print("Found 1 figure.")
        binary_image = np.where(image == image.max(), 1, 0)
        binary_image = binary_image.astype(np.uint8)
        binary_image = np.expand_dims(binary_image, axis=-1)
        if output_path is not None:
            save_path = os.path.join(output_path, f"{name}_new_serment.nrrd")
            nrrd.write(save_path, binary_image)
        else:
            nrrd.write(f"{name}_new_serment.nrrd", binary_image)

