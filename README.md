# data_analyst
This repository was created for sourcing , processing and analyzing any data 
##Data Analyst 
This project is a web-based data analyst agent built with Flask and the Gemini API. It can process and analyze data from various file formats, including text, CSV, JSON, and images, in response to a set of questions provided by the user.

##Features
###Multi-File Processing: Accepts multiple files in a single request, including .txt, .csv, .json, and image files (.png, .jpg, .jpeg).

###Gemini API Integration: Utilizes the Gemini API to analyze the provided data context and answer questions.

##External OCR Integration: Uses an external OCR service (like OCR.space) to extract text from image files, which is then sent to the Gemini model for analysis.

###JSON Output: The agent's response is a clean, structured JSON array, making it easy for other applications to parse and use.

##Prerequisites
Before you begin, ensure you have the following installed on your machine:

Python 3.8 or higher

pip (Python package installer)

##Setup and Installation
Clone the Repository:

git clone https://github.com/Ananthavalli1991/data_analyst.git
cd data_analyst

##Create a Virtual Environment:

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

##Install Dependencies:

pip install -r requirements.txt

##Configure Environment Variables:
Create a .env file in the root directory of your project with your API keys. Do not commit this file to Git.

GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
OCR_API_KEY="YOUR_OCR_API_KEY"

##Usage
Run the Application:
Start the Flask development server from the terminal.

python3 app.py

The application will run on http://localhost:5000.

##Prepare Files:
Create a questions.txt file and any data files (data.csv, image.png, etc.) you want to analyze in the same directory.

##Example questions.txt:

Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.

##Send the Request:
Use curl to send a POST request with your files to the API endpoint.

###To send only questions.txt:

curl -X POST "http://localhost:5000/api/" -F "questions.txt=@questions.txt"

###To send multiple files:

curl -X POST "http://localhost:5000/api/" -F "questions.txt=@questions.txt" -F "image.png=@image.png" -F "data.csv=@data.csv"

The API will return a JSON array with the answers to your questions.

##Deployment
This application is configured for easy deployment on platforms like Render. For detailed instructions, refer to the provided render.yaml, start.sh, and requirements.txt files in the repository.

##License
This project is licensed under the MIT License - see the LICENSE file for details.
