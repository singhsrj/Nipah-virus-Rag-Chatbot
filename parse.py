import os
from pathlib import Path

from llama_cloud import LlamaCloud
from dotenv import load_dotenv

load_dotenv()

client = LlamaCloud(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY")
)

corpus_dir = Path("corpus")
output_dir = Path("Preprocessed Files")

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# Get all PDFs
pdf_files = list(corpus_dir.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.\n")

for pdf_path in pdf_files:
    print(f"Processing: {pdf_path.name}")

    try:
        # Upload
        file_obj = client.files.create(
            file=str(pdf_path),
            purpose="parse"
        )

        # Parse
        result = client.parsing.parse(
            file_id=file_obj.id,
            tier="fast",
            version="latest",
            expand=["markdown_full", "text_full"],
        )

        # Use PDF filename without extension
        filename = pdf_path.stem

        # Save markdown
        markdown_path = output_dir / f"{filename}.md"
        markdown_path.write_text(
            result.markdown_full or "",
            encoding="utf-8"
        )

        # Save text
        text_path = output_dir / f"{filename}.txt"
        text_path.write_text(
            result.text_full or "",
            encoding="utf-8"
        )

        print(f"  ✓ Saved: {markdown_path}")
        print(f"  ✓ Saved: {text_path}\n")

    except Exception as e:
        print(f"  ✗ Failed: {pdf_path.name}")
        print(f"    Error: {e}\n")

print("Done!")