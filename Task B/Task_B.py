import cv2
import numpy as np
import os

# Removes table regions from a binary image based on pixel density analysis
def remove_table(binary_image, table_density_threshold=0.8):
    height, width = binary_image.shape
    horizontal_histogram = np.sum(binary_image == 255, axis=1)
    table_rows = np.where(horizontal_histogram > table_density_threshold * width)[0]
    table_mask = np.zeros_like(binary_image)
    if len(table_rows) > 0:
        table_mask[table_rows.min():table_rows.max(), :] = 255
    non_table_image = cv2.bitwise_and(binary_image, cv2.bitwise_not(table_mask))
    return non_table_image, table_mask

# Detects text columns in a binary image based on vertical whitespace
def column_detection(binary_image, min_column_width=100):
    vertical_histogram = np.sum(binary_image == 255, axis=0)
    column_gaps = np.where(vertical_histogram == 0)[0]
    column_boundaries = []
    prev_gap = 0
    for gap in column_gaps:
        if gap - prev_gap > min_column_width:
            column_boundaries.append((prev_gap, gap))
        prev_gap = gap
    if prev_gap < binary_image.shape[1]:
        column_boundaries.append((prev_gap, binary_image.shape[1]))
    return column_boundaries

# Detects paragraphs within a column by analyzing vertical spacing between text lines
def paragraph_detection(column_image, max_gap=40, table_mask=None):
    height, _ = column_image.shape
    horizontal_histogram = np.sum(column_image == 255, axis=1)
    text_rows = np.where(horizontal_histogram > 0)[0]
    paragraphs = []
    current_paragraph = []
    for i in range(len(text_rows)):
        if not current_paragraph:
            current_paragraph.append(text_rows[i])
        else:
            gap = text_rows[i] - text_rows[i - 1]
            if gap <= max_gap:
                current_paragraph.append(text_rows[i])
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [text_rows[i]]
    if current_paragraph:
        paragraphs.append(current_paragraph)
    refined_paragraphs = []
    for paragraph in paragraphs:
        top = paragraph[0]
        bottom = paragraph[-1]
        if table_mask is not None:
            table_intersection = np.any(table_mask[max(0, top - 20):min(height, bottom + 20), :] == 255)
            if table_intersection:
                continue
        refined_paragraphs.append((max(0, top - 20), min(height, bottom + 20)))
    return refined_paragraphs

# Main function to process multiple images, detecting and extracting paragraphs from columns.
def extract_paragraphs(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read the image at {image_path}")
            continue
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        non_table_image, table_mask = remove_table(binary_image)

        # --- Histogram projection visualizations ---
        # Row histogram: number of black pixels per row
        row_hist = np.sum(non_table_image == 255, axis=1)
        row_hist_img = np.ones((non_table_image.shape[0], 300, 3), dtype=np.uint8) * 255
        for y, val in enumerate(row_hist):
            cv2.line(row_hist_img, (0, y), (min(299, int(val)), y), (0, 0, 0), 1)
        row_hist_path = os.path.join(output_dir, f"{base_name}_row_histogram.png")
        cv2.imwrite(row_hist_path, row_hist_img)

        # Column histogram: number of black pixels per column
        col_hist = np.sum(non_table_image == 255, axis=0)
        col_hist_img = np.ones((300, non_table_image.shape[1], 3), dtype=np.uint8) * 255
        for x, val in enumerate(col_hist):
            cv2.line(col_hist_img, (x, 299), (x, max(0, 299 - int(val)),), (0, 0, 0), 1)
        col_hist_path = os.path.join(output_dir, f"{base_name}_column_histogram.png")
        cv2.imwrite(col_hist_path, col_hist_img)

        column_boundaries = column_detection(non_table_image)
        # Sort columns left to right
        column_boundaries.sort()
        for col_index, (start_col, end_col) in enumerate(column_boundaries):
            column_image = non_table_image[:, max(0, start_col - 20):min(non_table_image.shape[1], end_col + 20)]
            col_table_mask = table_mask[:, max(0, start_col - 20):min(non_table_image.shape[1], end_col + 20)]
            paragraphs = paragraph_detection(column_image, table_mask=col_table_mask)
            # Sort paragraphs top to bottom
            for para_index, (top, bottom) in enumerate(paragraphs):
                cropped_paragraph = column_image[top:bottom, :]
                paragraph_height, paragraph_width = cropped_paragraph.shape
                horizontal_histogram = np.sum(cropped_paragraph == 255, axis=1)
                table_row_indices = np.where(horizontal_histogram > 0.8 * paragraph_width)[0]
                if len(table_row_indices) > 0:
                    continue
                if paragraph_height < 40 or paragraph_width < 40:
                    continue
                cropped_paragraph = cv2.bitwise_not(cropped_paragraph)
                output_path = os.path.join(output_dir, f"{base_name}_column_{col_index + 1}_paragraph_{para_index + 1}.png")
                cv2.imwrite(output_path, cropped_paragraph)
        print(f"Processed {base_name} with {len(column_boundaries)} columns detected. Saved histogram projections.")

def process_images():
    base_dir = r"c:\Users\User\Desktop\CS Y2S1\CSC2014_FinalAssignment\CSC2014"
    input_dir = os.path.join(base_dir, "Converted Paper (8)")
    output_dir = os.path.join(base_dir, "Output", "Task B")
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    image_paths.sort()
    extract_paragraphs(image_paths, output_dir)

if __name__ == "__main__":
    process_images()
