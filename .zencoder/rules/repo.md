---
description: Repository Information Overview
alwaysApply: true
---

# Nivaratechai CRM Information

## Summary
A simple Flask-based CRM application with user authentication functionality. The application provides a login system with session management and a protected dashboard page that could be extended for customer relationship management features.

## Structure
- `app.py`: Main Flask application with user authentication logic
- `templates/`: HTML templates for the user interface
  - `login.html`: User login interface
  - `dashboard.html`: Protected dashboard page (potential CRM interface)
- `requirements.txt`: Python dependencies

## Language & Runtime
**Language**: Python
**Version**: Python 3.x
**Framework**: Flask 2.3.3
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- flask==2.3.3: Web framework for building the application
- flask-bcrypt==1.0.1: Password hashing for secure authentication

## Build & Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the application
python app.py

# Access the application at http://127.0.0.1:5000
# Default credentials:
# - Email: user@example.com
# - Password: password123
```

## CRM Features
**Current Authentication System**:
- Email and password-based authentication
- Password hashing with bcrypt
- Session management for persistent login
- Logout functionality

**CRM Potential**:
- Dashboard template ready for customer management interface
- User authentication system suitable for role-based access control
- Session tracking for user activity monitoring

## Development Roadmap
As indicated in the README, the following enhancements would transform this into a full CRM:
1. Database integration for customer data storage
2. User registration and management system
3. Customer information management features
4. Reporting and analytics capabilities
5. Enhanced security features (CSRF protection, rate limiting)