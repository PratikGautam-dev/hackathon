import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def get_crop_schedule(crop_name: str, state: str, season: str):
    try:
        if not GEMINI_API_KEY:
            raise ValueError("Missing GEMINI_API_KEY")

        prompt = f"""
        Create a structured 3-month task schedule for growing {crop_name} in {state} during {season} season.
        Format the response as a JSON array with tasks containing:
        - week_number (1-12 for 3 months)
        - task_description
        - category (soil_preparation/planting/irrigation/fertilizer/pest_management)
        """
        
        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("No response from Gemini API")
            
        try:
            # Clean and parse the response
            cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
            tasks = json.loads(cleaned_text)
            
            # Validate task structure
            for task in tasks:
                if not all(k in task for k in ('week_number', 'task_description', 'category')):
                    raise ValueError("Invalid task structure in response")
                
            return {
                "crop": crop_name,
                "state": state,
                "season": season,
                "tasks": tasks
            }
        except json.JSONDecodeError:
            # Fallback to basic task list
            tasks = [
                {
                    "week_number": i,
                    "task_description": f"Generic farming task for week {i}",
                    "category": "general"
                } for i in range(1, 13)
            ]
            return {
                "crop": crop_name,
                "state": state,
                "season": season,
                "tasks": tasks
            }

    except Exception as e:
        print(f"Schedule generation error: {str(e)}")
        return None
