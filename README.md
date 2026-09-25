# SmartResults 📚

SmartResults is a lightweight streamlit tool that automatically reads exam score sheets and extracts the numbers that matter (average grade, highest grade, lowest grade, average score, highest score, lowest score etc.).

## Dependencies

```
Package        Version
-------------- -------
google-genai   2.25.0
openai         3.19.2
pip            26.2.1
pypdf          6.19.0
python-dotenv  1.2.3
streamlit-chat 0.1.1
```

## Installation

1. Clone the repository ``git@github.com:AndrejGockov/SmartResults.git``
2. Create a .env file and add the environment variables: 
```
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GROQ_API_KEY=YOUR_GROQ_API_KEY
```
3. Run ``streamlit run app.py``

## License

This project is licensed under the terms of the [MIT license](LICENSE).