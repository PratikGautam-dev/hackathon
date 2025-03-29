# KrishiAI - Smart Farming Assistant

An AI-powered dashboard to help farmers make data-driven decisions for better crop management.

## Features

- Real-time field monitoring with NDVI tracking
- AI-driven recommendations for irrigation and fertilization
- Pest control alerts and management
- WhatsApp integration for instant notifications
- Multi-language support (English, Hindi, Tamil)

## Setup

1. Clone the repository
```bash
git clone <your-repo-url>
cd hackathon
```

2. Create virtual environment
```bash
python -m venv project-env
source project-env/bin/activate  # On Windows: project-env\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

## Environment Variables

Create a `.env` file with:
```
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
