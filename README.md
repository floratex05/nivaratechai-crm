sudo dnf install -y gcc gcc-c++ make python3-devel openblas-devel lapack-devel
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# Flask Login Page

A simple Flask application with a login page.

## Features

- User login with email and password
- Password hashing with bcrypt
- Session management
- Protected dashboard page
- Logout functionality

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`
3. Login with the following credentials:
   - Email: user@example.com
   - Password: password123

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
  - `login.html`: Login page template
  - `dashboard.html`: Dashboard page template (protected)
- `requirements.txt`: List of required packages

## Next Steps

In a real application, you would want to:
1. Use a database to store user information
2. Implement user registration
3. Add password reset functionality
4. Implement email verification
5. Add more security features (CSRF protection, rate limiting, etc.)