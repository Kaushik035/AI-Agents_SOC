1. Create and activate a virtual environment:

> pipenv shell

2. Install dependencies:

> pip install -r requirements.txt

3. Install the shared package in editable mode:(This makes the shared/ module directly available to your script—no need for PYTHONPATH)

> pip install -e .

4. Create an .env file in the root directory and add your proxy URL to the .env file:

> PROXY_URL=https://socapi.deepaksilaych.me/student1

5. From the project root, run it:

> python studyBuddy/scripts/study_buddy_week1.py (can directly run using the Run button in vs code)



🧪 Testing
Here’s how to test:

✅ 1. Clearly in notes

What is the history of the Taj Mahal?

✅ 2. Partially in notes

Who commissioned the Taj Mahal?

✅ 3. Not in notes

What is the capital of France?


If the answer is not in the notes, the assistant will say: I cannot find the answer in the provided notes.


