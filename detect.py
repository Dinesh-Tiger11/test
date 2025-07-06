# %%
import fitz  # PyMuPDF
import os
import cv2 # OpenCV for image processing (install with: pip install opencv-python)
import numpy as np # NumPy (often used with OpenCV)
import pandas as pd # Pandas for data manipulation (install with: pip install pandas)

def is_graph(image_path):
    """
    Placeholder function to determine if an image is a graph.
    This version also counts horizontal, vertical lines, and their intersections,
    with a minimum length requirement for lines.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing (bool, float or None, int, int, int).
               - bool: True if the image is likely a graph, False otherwise.
               - float: The calculated edge density, or None if an error occurred.
               - int: Number of horizontal lines detected.
               - int: Number of vertical lines detected.
               - int: Number of intersections between horizontal and vertical lines.
    """
    try:
        # Load the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Warning: Could not load image {image_path} for graph detection.")
            return False, None, 0, 0, 0

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get image dimensions for length-based filtering
        img_height, img_width = gray.shape[:2]

        # Apply Canny edge detector
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Calculate the percentage of edge pixels (edge density)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # For demonstration, a very basic, unreliable heuristic for graph detection:
        # If edge density is above a certain threshold, consider it a graph.
        # This threshold (0.05) is arbitrary and may need adjustment.
        is_it_graph = edge_density > 0.05

        # --- Line Detection using Hough Transform ---
        # HoughLinesP (Probabilistic Hough Line Transform) is used for better performance
        # and direct line segments.
        # Parameters:
        #   rho: Distance resolution of the accumulator in pixels.
        #   theta: Angle resolution of the accumulator in radians.
        #   threshold: Minimum number of votes (intersections in Hough grid cell).
        #   minLineLength: Minimum length of line. Line segments shorter than this are rejected.
        #   maxLineGap: Maximum allowed gap between line segments to treat them as single line.
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line length
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Calculate angle of the line in degrees
                # Avoid division by zero for vertical lines
                if x2 - x1 == 0:
                    angle_deg = 90
                else:
                    angle_rad = np.arctan2(y2 - y1, x2 - x1)
                    angle_deg = np.abs(np.degrees(angle_rad)) % 180 # Normalize to 0-180

                # Classify as horizontal or vertical based on angle and length
                # Only count if length is at least 50% of the image's relevant dimension
                if ((angle_deg < 5) or (angle_deg > 175)) and (line_length >= 0.5 * img_width): # Close to 0 or 180 degrees
                    horizontal_lines.append(((x1, y1), (x2, y2)))
                elif ((angle_deg > 85) and (angle_deg < 95)) and (line_length >= 0.5 * img_height): # Close to 90 degrees
                    vertical_lines.append(((x1, y1), (x2, y2)))

        num_horizontal_lines = len(horizontal_lines)
        num_vertical_lines = len(vertical_lines)

        # --- Intersection Counting ---
        intersection_count = 0
        for h_line in horizontal_lines:
            hx1, hy1 = h_line[0]
            hx2, hy2 = h_line[1]
            # Ensure hx1 is the smaller x-coordinate for consistent range checking
            hx_min, hx_max = min(hx1, hx2), max(hx1, hx2)
            
            for v_line in vertical_lines:
                vx1, vy1 = v_line[0]
                vx2, vy2 = v_line[1]
                # Ensure vy1 is the smaller y-coordinate for consistent range checking
                vy_min, vy_max = min(vy1, vy2), max(vy1, vy2)

                # An intersection occurs if the horizontal line's y-coordinate (hy1)
                # is within the vertical line's y-range (vy_min to vy_max) AND
                # the vertical line's x-coordinate (vx1) is within the horizontal line's x-range (hx_min to hx_max).
                # The condition is now on a single line to avoid potential syntax issues.
                if (vy_min <= hy1 <= vy_max) and (hx_min <= vx1 <= hx_max):
                    intersection_count += 1

        return is_it_graph, edge_density, num_horizontal_lines, num_vertical_lines, intersection_count

    except Exception as e:
        print(f"  Error during graph detection for {image_path}: {e}")
        return False, None, 0, 0, 0

def extract_images_from_pdf(pdf_path, output_base_folder="extracted_images"):
    """
    Extracts all images from a PDF document, attempts to classify them
    as graphs or other images, saves them to respective subfolders,
    and records image data (including edge density, line counts, intersections)
    to a CSV file.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_base_folder (str): The base folder where extracted images will be saved.
    """
    try:
        doc = fitz.open(pdf_path)
        print(f"Opened PDF: {pdf_path}")

        # Create base output folder
        if not os.path.exists(output_base_folder):
            os.makedirs(output_base_folder)
            print(f"Created base output folder: {output_base_folder}")

        # Create subfolders for graphs and other images
        graphs_folder = os.path.join(output_base_folder, "graphs")
        other_images_folder = os.path.join(output_base_folder, "other_images")

        if not os.path.exists(graphs_folder):
            os.makedirs(graphs_folder)
            print(f"Created graphs folder: {graphs_folder}")
        if not os.path.exists(other_images_folder):
            os.makedirs(other_images_folder)
            print(f"Created other images folder: {other_images_folder}")

        image_count = 0
        graph_count = 0
        other_count = 0
        image_data_for_csv = [] # List to store data for the pandas DataFrame

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            print(f"Processing page {page_num + 1}...")

            image_list = page.get_images(full=True)

            if not image_list:
                print(f"No images found on page {page_num + 1}.")
                continue

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Temporarily save the image to check if it's a graph
                temp_image_filename = os.path.join(
                    output_base_folder,
                    f"temp_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                )
                with open(temp_image_filename, "wb") as img_file:
                    img_file.write(image_bytes)

                # Determine if it's a graph and get all new metrics
                is_it_graph, current_edge_density, \
                horiz_lines, vert_lines, intersections = is_graph(temp_image_filename)

                # Define destination folder and classification string
                if is_it_graph:
                    destination_folder = graphs_folder
                    classification_str = "Graph"
                    graph_count += 1
                else:
                    destination_folder = other_images_folder
                    classification_str = "Other Image"
                    other_count += 1

                # Construct the final image filename
                final_image_filename = os.path.join(
                    destination_folder,
                    f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                )

                # Move the image to its final destination
                os.rename(temp_image_filename, final_image_filename)
                print(f"  Classified as {classification_str}: {os.path.basename(final_image_filename)}")
                print(f"  Saved to: {final_image_filename}")
                image_count += 1

                # Store data for CSV
                image_data_for_csv.append({
                    "Image Name": os.path.basename(final_image_filename),
                    "Page Number": page_num + 1,
                    "Edge Density": current_edge_density,
                    "Horizontal Lines": horiz_lines,
                    "Vertical Lines": vert_lines,
                    "Intersections": intersections,
                    "Classification": classification_str
                })

        doc.close()
        print(f"\nImage extraction and classification complete.")
        print(f"Total images extracted: {image_count}")
        print(f"Graphs identified (heuristic): {graph_count}")
        print(f"Other images identified: {other_count}")
        print(f"Images saved to: {os.path.abspath(output_base_folder)}")

        # Create a pandas DataFrame and save to CSV
        if image_data_for_csv:
            df = pd.DataFrame(image_data_for_csv)
            csv_path = os.path.join(output_base_folder, "image_classification_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nImage classification results saved to: {os.path.abspath(csv_path)}")
        else:
            print("\nNo image data to save to CSV.")

    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- How to use the function ---
if __name__ == "__main__":
    # IMPORTANT: Replace 'your_document.pdf' with the actual path to your PDF file
    pdf_file = "HT_understand_graphs.pdf"
    output_directory = "extracted_content" # Base folder for all extracted images
    extract_images_from_pdf(pdf_file, output_directory)

    print("\nNote: The graph detection logic in 'is_graph()' is a simplified placeholder.")
    print("For accurate graph detection, you would need to implement more advanced")
    print("computer vision techniques (e.g., using OpenCV for feature detection,")
    print("or machine learning models trained on graph datasets).")
    print("The line and intersection counting is also a basic implementation and may")
    print("require tuning of Hough Transform parameters for your specific images.")

# %%
