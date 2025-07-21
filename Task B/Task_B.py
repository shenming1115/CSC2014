import cv2
import numpy as np
import os

# Removes table regions from a binary image based on pixel density analysis
# This function creates a mask for table-like regions (high pixel density rows) and removes them from the image.
def remove_table(binary_image, table_density_threshold=0.8):
    height, width = binary_image.shape
    horizontal_histogram = np.sum(binary_image == 255, axis=1)  # Count black pixels per row
    table_rows = np.where(horizontal_histogram > table_density_threshold * width)[0]  # Rows with high density
    table_mask = np.zeros_like(binary_image)
    if len(table_rows) > 0:
        table_mask[table_rows.min():table_rows.max(), :] = 255  # Mark table region in mask
    non_table_image = cv2.bitwise_and(binary_image, cv2.bitwise_not(table_mask))  # Remove table region
    return non_table_image, table_mask

# Detects text columns in a binary image based on vertical whitespace
# This function finds column boundaries by locating vertical gaps (no black pixels) and grouping columns.
def column_detection(binary_image, min_column_width=100):
    vertical_histogram = np.sum(binary_image == 255, axis=0)  # Count black pixels per column
    column_gaps = np.where(vertical_histogram == 0)[0]        # Columns with no black pixels (gaps)
    column_boundaries = []
    prev_gap = 0
    for gap in column_gaps:
        if gap - prev_gap > min_column_width:
            column_boundaries.append((prev_gap, gap))  # Add column boundary if wide enough
        prev_gap = gap
    if prev_gap < binary_image.shape[1]:
        column_boundaries.append((prev_gap, binary_image.shape[1]))  # Add last column
    return column_boundaries

# Detects paragraphs within a column by analyzing vertical spacing between text lines
# This function groups consecutive text rows into paragraphs, skipping table regions if detected.
def paragraph_detection(column_image, max_gap=40, table_mask=None):
    height, _ = column_image.shape
    horizontal_histogram = np.sum(column_image == 255, axis=1)  # Count black pixels per row
    text_rows = np.where(horizontal_histogram > 0)[0]           # Rows with text
    paragraphs = []
    current_paragraph = []
    for i in range(len(text_rows)):
        if not current_paragraph:
            current_paragraph.append(text_rows[i])
        else:
            gap = text_rows[i] - text_rows[i - 1]
            if gap <= max_gap:
                current_paragraph.append(text_rows[i])  # Continue current paragraph
            else:
                paragraphs.append(current_paragraph)    # Start new paragraph
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
                continue  # Skip paragraphs overlapping with table
        refined_paragraphs.append((max(0, top - 20), min(height, bottom + 20)))  # Add paragraph bounds
    return refined_paragraphs

# Main function to process multiple images, detecting and extracting paragraphs from columns.
# This function processes each input image, generates histogram visualizations, detects columns and paragraphs, and saves results.
def extract_paragraphs(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # Get image filename (without extension)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)           # Read image in grayscale
        if image is None:
            print(f"Could not read the image at {image_path}")         # Print error if image not found
            continue
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # Binarize image (black text = 255)
        non_table_image, table_mask = remove_table(binary_image)                  # Remove table regions

        # --- Histogram projection visualizations ---
        # Row histogram: number of black pixels per row
        row_hist = np.sum(non_table_image == 255, axis=1)
        row_hist_img = np.ones((non_table_image.shape[0], 300, 3), dtype=np.uint8) * 255  # White canvas for row histogram
        for y, val in enumerate(row_hist):
            cv2.line(row_hist_img, (0, y), (min(299, int(val)), y), (0, 0, 0), 1)        # Draw black line for each row
        row_hist_path = os.path.join(output_dir, f"{base_name}_row_histogram.png")
        cv2.imwrite(row_hist_path, row_hist_img)                                         # Save row histogram image

        # Column histogram: number of black pixels per column
        col_hist = np.sum(non_table_image == 255, axis=0)
        col_hist_img = np.ones((300, non_table_image.shape[1], 3), dtype=np.uint8) * 255  # White canvas for column histogram
        for x, val in enumerate(col_hist):
            cv2.line(col_hist_img, (x, 299), (x, max(0, 299 - int(val)),), (0, 0, 0), 1) # Draw black line for each column
        col_hist_path = os.path.join(output_dir, f"{base_name}_column_histogram.png")
        cv2.imwrite(col_hist_path, col_hist_img)                                         # Save column histogram image

        column_boundaries = column_detection(non_table_image)                            # Detect columns
        # Sort columns left to right
        column_boundaries.sort()
        for col_index, (start_col, end_col) in enumerate(column_boundaries):
            column_image = non_table_image[:, max(0, start_col - 20):min(non_table_image.shape[1], end_col + 20)]  # Crop column
            col_table_mask = table_mask[:, max(0, start_col - 20):min(non_table_image.shape[1], end_col + 20)]     # Crop table mask for column
            paragraphs = paragraph_detection(column_image, table_mask=col_table_mask)                              # Detect paragraphs in column
            # Sort paragraphs top to bottom
            for para_index, (top, bottom) in enumerate(paragraphs):
                cropped_paragraph = column_image[top:bottom, :]                                                  # Crop paragraph region
                paragraph_height, paragraph_width = cropped_paragraph.shape
                horizontal_histogram = np.sum(cropped_paragraph == 255, axis=1)                                  # Histogram for paragraph
                table_row_indices = np.where(horizontal_histogram > 0.8 * paragraph_width)[0]                    # Check for table rows
                if len(table_row_indices) > 0:
                    continue  # Skip if paragraph overlaps with table
                if paragraph_height < 40 or paragraph_width < 40:
                    continue  # Skip small regions
                cropped_paragraph = cv2.bitwise_not(cropped_paragraph)                                           # Invert for output
                output_path = os.path.join(output_dir, f"{base_name}_column_{col_index + 1}_paragraph_{para_index + 1}.png")
                cv2.imwrite(output_path, cropped_paragraph)                                                      # Save paragraph image
        print(f"Processed {base_name} with {len(column_boundaries)} columns detected. Saved histogram projections.")

# Main entry point: prepares input/output paths and runs the extraction process.
def process_images():
    base_dir = r"c:\Users\User\Desktop\CS Y2S1\CSC2014_FinalAssignment\CSC2014"  # Base directory for project
    input_dir = os.path.join(base_dir, "Converted Paper (8)")                         # Input images directory
    output_dir = os.path.join(base_dir, "Output", "Task B")                          # Output directory for results
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.png')]  # List all PNG images
    image_paths.sort()                                                                 # Sort images for consistent order
    extract_paragraphs(image_paths, output_dir)                                        # Run extraction and visualization

if __name__ == "__main__":
    process_images()  # Run main process if script is executed directly
