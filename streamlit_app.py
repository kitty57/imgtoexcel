import json
import time
import base64
import warnings
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import io
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

st.title("🖼 OCR : Image Text Extraction")


def putMarkdown():
    st.markdown("<hr>", unsafe_allow_html=True)


def get_download_button(data, button_text, filename):
    json_str = json.dumps(data, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;charset=utf-8;base64,{b64}" download="{filename}">{button_text}</a>'
    return href


def ocr(item):
    model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
    result = model(item)
    json_output = result.export()
    return result, json_output


def annotate_image(img, json_output):
    """Draw bounding boxes and text on the image."""
    img = np.array(img)  # Convert to numpy array (if using PIL)

    for block in json_output["pages"][0]["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                (x, y, w, h) = (
                    int(word["geometry"][0][0] * img.shape[1]),
                    int(word["geometry"][0][1] * img.shape[0]),
                    int(word["geometry"][1][0] * img.shape[1]),
                    int(word["geometry"][1][1] * img.shape[0]),
                )
                text = word["value"]

                # Draw bounding box
                cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)

                # Put text
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img


def create_improved_table_layout(json_output):
    """
    Create a table layout from OCR results that preserves the spatial arrangement
    of text as it appears in the original image, with improved column handling.
    """
    # Early validation of input
    if not json_output or not isinstance(json_output, dict):
        st.error("Invalid JSON output received")
        return pd.DataFrame()

    # Check if pages exist and are not empty
    if "pages" not in json_output or not json_output["pages"]:
        st.error("No pages found in the JSON output")
        return pd.DataFrame()

    # Validate the first page structure
    first_page = json_output["pages"][0]
    if "blocks" not in first_page or not first_page["blocks"]:
        st.error("No blocks found in the first page")
        return pd.DataFrame()

    # Extract all words with their positions
    words_data = []
    for block in first_page["blocks"]:
        for line in block.get("lines", []):
            for word in line.get("words", []):
                # Validate word geometry
                if "geometry" not in word or len(word["geometry"]) < 2:
                    continue

                # Get coordinates (normalized)
                try:
                    x_min = float(word["geometry"][0][0])
                    y_min = float(word["geometry"][0][1])
                    x_max = float(word["geometry"][1][0])
                    y_max = float(word["geometry"][1][1])
                except (ValueError, IndexError, TypeError):
                    continue

                # Use top-left corner for better alignment in tables
                words_data.append({
                    "text": word.get("value", ""),
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "height": y_max - y_min,
                    "width": x_max - x_min
                })

    # Check if any words were extracted
    if not words_data:
        st.error("No words could be extracted from the image")
        return pd.DataFrame()

    # Create DataFrame from extracted words
    df = pd.DataFrame(words_data)

    # Improved row and column detection
    def cluster_coordinates(coords, max_distance):
        """Group coordinates that are within max_distance of each other"""
        if len(coords) <= 1:
            return [0] * len(coords)

        # Sort coordinates
        sorted_indices = np.argsort(coords)
        sorted_coords = coords[sorted_indices]

        # Find gaps larger than max_distance
        diffs = np.diff(sorted_coords)
        cluster_breaks = np.where(diffs > max_distance)[0]

        # Assign cluster IDs
        cluster_ids = np.zeros(len(coords), dtype=int)
        for i, break_idx in enumerate(cluster_breaks):
            cluster_ids[sorted_indices[break_idx + 1:]] = i + 1

        return cluster_ids[np.argsort(sorted_indices)]

    # Get typical line height as clustering parameter
    # Add error handling for mean calculation
    try:
        mean_height = df["height"].mean()
        row_threshold = mean_height * 0.5
    except Exception as e:
        st.error(f"Error calculating row threshold: {e}")
        mean_height = 0.1
        row_threshold = 0.05

    # Cluster by y_min coordinates (rows)
    y_coords = df["y_min"].values
    df["row"] = cluster_coordinates(y_coords, row_threshold)

    # Determine column positions more robustly
    def get_column_positions(df, num_expected_cols=10):
        """Identify potential column positions across all rows"""
        try:
            # Group x_min coordinates across different rows
            x_min_coords = df.groupby("row")["x_min"].apply(list).explode().values

            # Avoid clustering if not enough coordinates
            if len(x_min_coords) < 3:
                st.error("Not enough coordinates for clustering")
                return []

            # Use K-means to cluster column positions
            # Reshape coordinates for clustering
            X = x_min_coords.reshape(-1, 1)

            # Use a range of clusters around the expected number of columns
            n_clusters = max(3, min(num_expected_cols, len(X)))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            kmeans.fit(X)

            # Sort cluster centers to represent column positions
            column_centers = sorted(kmeans.cluster_centers_.flatten())

            return column_centers
        except Exception as e:
            st.error(f"Error in column position clustering: {e}")
            return []

    # Get column positions with error handling
    column_positions = get_column_positions(df)

    # Assign columns based on x_min proximity to column centers
    def assign_column(x_min, column_positions):
        """Assign a word to the nearest column"""
        if not column_positions:
            return 0
        return np.argmin(np.abs(np.array(column_positions) - x_min))

    df["col"] = df["x_min"].apply(lambda x: assign_column(x, column_positions))

    # Sort by row and column
    df = df.sort_values(["row", "col"])

    # Create the table with placeholders for potentially missing columns
    # Get max row and column to determine table size
    max_row = df["row"].max()
    max_col = df["col"].max()

    # Create a table with all possible columns
    table_data = {}
    for row in range(max_row + 1):
        row_data = {}
        for col in range(max_col + 1):
            # Find words in this specific row and column
            cell_words = df[(df["row"] == row) & (df["col"] == col)]["text"].tolist()
            row_data[col] = ' '.join(cell_words) if cell_words else ''
        table_data[row] = row_data

    # Convert to DataFrame, filling missing columns
    table_df = pd.DataFrame.from_dict(table_data, orient='index')

    # Rename columns to be more descriptive
    table_df.columns = [f"Column {i + 1}" for i in range(len(table_df.columns))]

    # Sort by row index to maintain original order
    table_df = table_df.sort_index()

    return table_df


def display(result, json_output, img):
    st.write("#### Download JSON Output")
    st.write("⬇" * 9)

    download_button_str = get_download_button(json_output, "DOWNLOAD", "data.json")
    st.markdown(download_button_str, unsafe_allow_html=True)
    putMarkdown()

    st.image(img, caption="Original Image")
    putMarkdown()

    # Annotate the image with extracted text
    annotated_img = annotate_image(Image.open(img), json_output)
    st.image(annotated_img, caption="Annotated Image with Extracted Text")

    synthetic_pages = result.synthesize()
    st.image(synthetic_pages, caption="OCR Result")
    putMarkdown()

    # Extract all text for use in the table
    extracted_text = []
    for block in json_output["pages"][0]["blocks"]:
        for line in block["lines"]:
            line_text = []
            for word in line["words"]:
                line_text.append(word["value"])
            extracted_text.append(" ".join(line_text))

    st.write("## Extracted Text:")
    st.write("\n".join(extracted_text))
    putMarkdown()

    # Table Layout for Excel export
    st.write("## Table Layout (Copy-paste to Excel)")

    # Create improved table layout
    try:
        table_df = create_improved_table_layout(json_output)

        if not table_df.empty:
            # Display the table with a larger width for better visualization
            st.dataframe(table_df, use_container_width=True)

            # Add CSV download option for Excel import
            csv = table_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="extracted_table.csv">Download CSV for Excel</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Add Excel download option (XLSX format)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                table_df.to_excel(writer, index=False, sheet_name='Extracted Table')

            excel_data = buffer.getvalue()
            b64_excel = base64.b64encode(excel_data).decode()
            href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="extracted_table.xlsx">Download Excel File</a>'
            st.markdown(href_excel, unsafe_allow_html=True)

            # Add a copy button (using HTML/JavaScript)
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:10px;">
                <p>For best results when copying to Excel:</p>
                <ol>
                    <li>Click and drag to select all cells in the table above</li>
                    <li>Use Ctrl+C (or Cmd+C on Mac) to copy</li>
                    <li>Open Excel and use Ctrl+V (or Cmd+V) to paste</li>
                </ol>
                <p>Alternatively, download the Excel file using the link above.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("No structured table data could be extracted from this image.")
    except Exception as e:
        st.error(f"Error creating table layout: {str(e)}")
        st.write("Falling back to simpler table extraction method...")

        # Simple fallback method for table extraction
        try:
            # Create a simple table from text lines
            lines = []
            for block in json_output["pages"][0]["blocks"]:
                for line in block["lines"]:
                    line_text = []
                    for word in line["words"]:
                        line_text.append(word["value"])
                    lines.append(" ".join(line_text))

            # Create a single-column dataframe
            simple_df = pd.DataFrame(lines, columns=["Text"])
            st.dataframe(simple_df, use_container_width=True)

            # Add CSV download option
            csv = simple_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="extracted_text.csv">Download CSV for Excel</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e2:
            st.error(f"Error with fallback method: {str(e2)}")

    elapsed_time = time.time() - start_time
    putMarkdown()
    st.write(f"Successful! Passed Time: {elapsed_time:.2f} seconds")


def main():
    global start_time

    uploaded_file = st.file_uploader("Upload a File", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        start_time = time.time()

        if uploaded_file.type == "application/pdf":
            image = uploaded_file.read()
            single_img_doc = DocumentFile.from_pdf(image)
        else:
            image = uploaded_file.read()
            single_img_doc = DocumentFile.from_images(image)

        result, json_output = ocr(single_img_doc)
        display(result, json_output, uploaded_file)

main()
