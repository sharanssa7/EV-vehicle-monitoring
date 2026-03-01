# EV Health Monitoring System

## Setup in VS Code

1. Extract ZIP
2. Open folder in VS Code
3. Create virtual environment:

    python -m venv venv
    venv\Scripts\activate

4. Install dependencies:

    pip install -r requirements.txt

5. Run server:

    uvicorn app:app --reload

6. Open browser:

    http://127.0.0.1:8000/docs
