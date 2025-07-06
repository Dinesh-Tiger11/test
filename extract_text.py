import os
import json
import boto3
import base64
from pdf2image import convert_from_path
from trp import Document
from PIL import Image
from io import BytesIO

# === CONFIGURATION ===
pdf_path = "/home/ec2-user/SageMaker/your-file.pdf"  # <-- Update this
image_dir = "/home/ec2-user/SageMaker/pdf_images"
output_path = "/home/ec2-user/SageMaker/structured_output.json"
os.makedirs(image_dir, exist_ok=True)

# Initialize AWS Textract
textract = boto3.client("textract")

# Final output list
document_structure = []

# === STEP 1: Convert PDF to Images ===
images = convert_from_path(pdf_path, dpi=300)
print(f"Converted PDF into {len(images)} page image(s).")

# === STEP 2: Process each page ===
for page_index, pil_image in enumerate(images):
    page_number = page_index + 1
    width, height = pil_image.size
    local_id = 1
    page_elements = []

    print(f"\n🔍 Processing Page {page_number}...")

    # Save image
    image_path = os.path.join(image_dir, f"page_{page_number}.png")
    pil_image.save(image_path, "PNG")

    # Read image bytes
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    # Call Textract
    response = textract.analyze_document(
        Document={"Bytes": image_bytes},
        FeatureTypes=["TABLES", "FORMS"]
    )

    # Parse response
    doc = Document(response)
    page = doc.pages[0]

    # 1. Extract text lines
    for line in page.lines:
        page_elements.append({
            "id": local_id,
            "type": "text",
            "top": line.geometry.boundingBox.top,
            "content": line.text.strip()
        })
        local_id += 1

    # 2. Extract tables (with headers and title detection)
    for table in page.tables:
        table_top = table.geometry.boundingBox.top
        table_bottom = table_top + table.geometry.boundingBox.height

        # Try to find a nearby title (above or below the table)
        title_candidates = []
        for line in page.lines:
            line_top = line.geometry.boundingBox.top
            distance_above = table_top - line_top
            distance_below = line_top - table_bottom

            if 0 < distance_above < 0.05:
                title_candidates.append((distance_above, line.text.strip()))
            elif 0 < distance_below < 0.05:
                title_candidates.append((distance_below, line.text.strip()))

        title_candidates.sort(key=lambda x: x[0])
        table_title = title_candidates[0][1] if title_candidates else None

        # Extract table rows
        rows = []
        for row in table.rows:
            row_data = [cell.text.strip() if cell.text else "" for cell in row.cells]
            rows.append(row_data)

        headers = rows[0] if len(rows) > 1 else []
        table_rows = rows[1:] if len(rows) > 1 else rows

        page_elements.append({
            "id": local_id,
            "type": "table",
            "top": table_top,
            "title": table_title,
            "headers": headers,
            "rows": table_rows
        })
        local_id += 1

    # 3. Extract other visual elements (images)
    for block in response['Blocks']:
        if block['BlockType'] not in ['LINE', 'TABLE', 'CELL', 'WORD']:
            bbox = block['Geometry']['BoundingBox']
            left = int(bbox['Left'] * width)
            top = int(bbox['Top'] * height)
            w = int(bbox['Width'] * width)
            h = int(bbox['Height'] * height)

            if w > 10 and h > 10:  # ignore tiny blocks
                cropped = pil_image.crop((left, top, left + w, top + h))
                buffered = BytesIO()
                cropped.save(buffered, format="PNG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

                page_elements.append({
                    "id": local_id,
                    "type": "image",
                    "top": bbox['Top'],
                    "content": encoded_image
                })
                local_id += 1

    # Sort all elements top-to-bottom
    page_elements.sort(key=lambda x: x["top"])
    for e in page_elements:
        e.pop("top", None)  # remove layout info from final output

    # Append to overall document structure
    document_structure.append({
        "page": page_number,
        "elements": page_elements
    })

# === STEP 3: Save Structured Output ===
with open(output_path, "w") as out_file:
    json.dump(document_structure, out_file, indent=2)

print(f"\n✅ Structured JSON saved to: {output_path}")

# Load Tables from Json
import json
import pandas as pd

# === Load the structured JSON ===
input_path = "/home/ec2-user/SageMaker/structured_output.json"  # update path if needed

with open(input_path, "r") as f:
    data = json.load(f)

# === Extract and display tables as DataFrames ===
table_count = 1
for page in data:
    page_num = page["page"]

    for element in page["elements"]:
        if element["type"] == "table":
            title = element.get("title", f"Table {table_count}")
            headers = element.get("headers", [])
            rows = element.get("rows", [])

            # Fallback if no headers
            if not headers and rows:
                headers = [f"Col{i+1}" for i in range(len(rows[0]))]

            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            print(f"\n📄 Page {page_num} | 📝 Table {table_count}: {title}")
            display(df)

            table_count += 1

# Display text and images
import json
import base64
from PIL import Image
from io import BytesIO

# === Load structured JSON ===
input_path = "/home/ec2-user/SageMaker/structured_output.json"  # adjust if needed

with open(input_path, "r") as f:
    data = json.load(f)

# === Step 1: Display all text (in order) ===
print("\n📝 ALL TEXT (Ordered):\n" + "=" * 50)

for page in data:
    print(f"\n📄 Page {page['page']}")
    for element in page["elements"]:
        if element["type"] == "text":
            print(element["content"])

# === Step 2: Show all images (in order) ===
print("\n🖼️ DISPLAYING IMAGES (in order)...")

image_count = 1
for page in data:
    for element in page["elements"]:
        if element["type"] == "image":
            img_data = base64.b64decode(element["content"])
            img = Image.open(BytesIO(img_data))
            print(f"\n📸 Image {image_count} (from Page {page['page']})")
            img.show()  # This will open the image in default viewer
            image_count += 1