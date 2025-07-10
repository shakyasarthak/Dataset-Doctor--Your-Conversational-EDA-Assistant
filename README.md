# Dataset Doctor - Your Conversational EDA Assistant

*A web-based chat bot that allows the user to have a conversation with their CSV files.*

## Overview

**Dataset Doctor** is an interactive, beginner-friendly Streamlit application for exploring, analyzing, cleaning, and visualizing CSV datasets—simply by chatting with your data. No coding experience is required. This app leverages advanced AI models to answer questions, generate visualizations, and provide automated insights.

## Features

- **Chat with Your Data:** Ask questions in plain English and get instant, meaningful answers.
- **CSV Upload & Preview:** Easily upload CSV files and preview your dataset.
- **Automated Data Cleaning:** Remove duplicates, fill in missing values, and handle outliers with a single click.
- **Visualizations:** Instantly generate histograms, scatter plots, bar charts, heatmaps, and more.
- **Exploratory Data Analysis (EDA) Reports:** Create comprehensive data analysis reports automatically.
- **Feature Engineering Suggestions:** Get ideas for creating new features from your data.
- **Download Cleaned Data:** Save your improved dataset for further use.
- **Export Chat History:** Keep a record of your analysis and conversations.
- **Security Warnings:** Clear notifications help you use the app safely.

## System Requirements

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Edge, etc.)

### Required Python Packages

- streamlit
- together
- ydata-profiling
- pandas
- plotly
- seaborn
- matplotlib
- python-dotenv

All dependencies are listed in the `requirements.txt` file.

## Step-by-Step Installation Guide

### 1. Clone the Repository

Open your terminal (Command Prompt, PowerShell, or Terminal app) and run:

```bash
git clone https://github.com/yourusername/dataset-doctor.git
cd dataset-doctor
```

### 2. (Optional) Create a Virtual Environment

This helps keep your project’s packages organized.

```bash
python -m venv venv
```

- **Windows:**  
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**  
  ```bash
  source venv/bin/activate
  ```

### 3. Install the Required Packages

```bash
pip install -r requirements.txt
```

## How to Get Your API Key

Dataset Doctor uses the Together AI API for language model features. You must obtain your own free API key:

1. Go to [Together AI](https://www.together.ai/).
2. Sign up for a free account.
3. After logging in, find your API key in your account dashboard.

> **Note:** Together AI offers a free tier, but it has usage limits and may require payment for heavier use. Suggestions or contributions to support other free and open-source AI models are welcome for future versions.

## Setting Up the Environment File

1. In your project folder, create a file named `.env` (just `.env`, no filename).
2. Open `.env` in a text editor and add your API key like this:

   ```
   TOGETHER_API_KEY=your_together_api_key_here
   ```

3. Save the file.

## How to Launch Dataset Doctor

In your terminal, run:

```bash
streamlit run app.py
```

- A local URL (like `http://localhost:8501`) will appear in your terminal.
- Open this URL in your web browser.

## How to Use Dataset Doctor

1. **Upload a CSV File:**  
   Click the "Upload CSV" button and select your data file.

2. **Preview Your Data:**  
   See the first few rows and columns instantly.

3. **Chat with Your Data:**  
   Type questions or commands in plain English. For example:
   - "Show histogram of Age"
   - "Find missing values"
   - "Generate EDA report"

4. **View Visualizations and Reports:**  
   The app creates charts and reports automatically based on your requests.

5. **Clean and Download Data:**  
   Use the sidebar tools to clean your data and download the improved dataset.

6. **Export Chat History:**  
   Save your conversation for documentation or reproducibility.

## Example Questions You Can Ask

- "Show histogram of Age"
- "What is the correlation between price and size?"
- "Generate EDA report"
- "Clean outliers in Price"
- "Suggest feature engineering"
- "Show bar chart of gender"
- "Describe the dataset"
- "Find columns with missing values"

## Dataset Requirements

- **Format:** CSV (Comma-Separated Values)
- **Columns:** Any tabular structure is supported. For best results, use descriptive column names and consistent data types.
- **Size:** Handles most small to medium datasets (performance may vary for very large files).

## Security Notice

> **Dataset Doctor executes Python code for data analysis.**
>
> For your safety:
> - Only upload data from trusted sources.
> - Review all code suggestions before executing.
> - Avoid using sensitive or private data.

## Open Source AI Model Suggestions

- Together AI is not fully free and has usage limits.
- Suggestions and contributions for integrating free and open-source AI models (such as Hugging Face Transformers, Llama.cpp, etc.) are encouraged to make Dataset Doctor even more accessible.

## Contributing & Suggestions

- Beginner-friendly contributions are welcome!
- If you have ideas for improving Dataset Doctor—especially for supporting free/open-source AI models—open an issue or submit a pull request on GitHub.
- For help, check the Issues section or leave your questions there.

**Enjoy exploring your data with Dataset Doctor!**

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/64888235/f073623c-7663-4310-8c1c-e458cc9e367b/app.py
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/64888235/2213ce7b-81f4-411f-a7bf-53eb58eb667f/chatbot.py
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/64888235/79301ca0-18f7-4fd9-a0ec-41b3976328a6/eda_engine.py
[4] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/64888235/b75eed0c-9d5d-4139-9ccc-51674e9b7b13/visualizer.py
