class Config:
    """Configuration settings for the application."""
    
    # Database Configuration
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = "Tharuicbt01"
    DB_NAME = "attendance_db"
    
    # Other configurations (if needed in the future)
    SECRET_KEY = "your_secret_key_here"  # Change this for security purposes

# Example usage:
if __name__ == "__main__":
    print("Database Host:", Config.DB_HOST)
    print("Database Name:", Config.DB_NAME)
