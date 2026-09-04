# Data with Roots

University Machine Learning project built as a Flask web application. It presents Machine
Learning concepts, real-world use cases, and a guided walkthrough of Linear Regression.

## Objective

Provide a clear, well-structured web app that explains Machine Learning fundamentals and
showcases a supervised learning algorithm (Linear Regression), combining an academic
presentation with a clean, modern design.

## Technologies

- Python
- Flask
- HTML5 + Jinja2
- Bootstrap 5
- Git / GitHub

## Project Structure

```text
data-with-roots/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   ├── base.html          # Shared layout: navbar, footer, blocks
│   ├── home.html          # Landing page
│   │
│   ├── ml/                # Machine Learning section
│   │   ├── concepts.html
│   │   └── types.html
│   │
│   ├── use_cases/         # Use Cases section
│   │   ├── use_case_1.html
│   │   ├── use_case_2.html
│   │   ├── use_case_3.html
│   │   └── use_case_4.html
│   │
│   └── regression/        # Supervised > Linear Regression
│       ├── concepts.html
│       └── application.html
│
└── static/
    ├── css/style.css
    ├── js/script.js
    └── images/
```

## Getting Started

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   - Windows: `venv\Scripts\activate`
   - macOS / Linux: `source venv/bin/activate`

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Open your browser at `http://127.0.0.1:5000`.

## Routes

| Route                    | Description                     |
|---------------------------|----------------------------------|
| `/`                        | Home                             |
| `/ml/concepts`             | Machine Learning Concepts        |
| `/ml/types`                | Types of Machine Learning        |
| `/use-cases/1..4`          | Use Cases 1 to 4                 |
| `/regression/concepts`     | Linear Regression - Concepts     |
| `/regression/application`  | Linear Regression - Application  |

## Status

The base structure, navigation, and Home page are complete. Sections marked as
"Coming Soon" are placeholders ready for the rest of the team to develop.
