# OritzPDF - Intelligent Document Assistant

An intelligent document assistant leveraging Claude Code to analyze, summarize, and interact with PDF and document files using natural language.

## Features

- **Text & Visual Element Analysis**: Extract insights from charts, tables, diagrams, and text
- **OCR Capabilities**: Process scanned documents
- **Summarization Engine**: Summarize entire documents or specific page ranges
- **Intelligent Q&A**: Answer content-based queries within documents
- **Structured Data Extraction**: Extract tables, metrics, legal clauses, and entities
- **Multilingual Translation**: Translate sections or whole documents on-demand
- **PDF Comparison Tool**: Highlight differences between multiple PDFs
- **Conversational Interface**: Natural language commands for document operations

## Supported Formats

- PDF (up to 100 pages or 32MB)
- DOCX, TXT, CSV, HTML, XLSX

## Project Structure

```
OritzPDF/
├── src/
│   ├── api/            # API endpoints
│   ├── services/       # Business logic and processing
│   ├── models/         # Data models and schemas
│   ├── utils/          # Utility functions
│   └── tests/          # Test files
├── config/             # Configuration files
├── docs/               # Documentation
├── scripts/            # Utility scripts
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variables template
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure your settings
5. Run the application:
   ```bash
   uvicorn src.main:app --reload
   ```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## Configuration

The application can be configured via environment variables. See `.env.example` for all available options.

Key configuration areas:
- **OCR Engine**: Choose between Tesseract (free) or cloud services (Google Vision, Azure)
- **Translation**: Configure Google Translate, DeepL, or Azure Translator
- **Storage**: Local filesystem or S3-compatible storage
- **NLP Models**: Configure summarization and Q&A models

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black src/
```

Lint code:
```bash
flake8 src/
```

## License

[Add your license here]