import mysql.connector
from mysql.connector import Error

def create_connection():
    """Establishes and returns a connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',  # Change if you have a different username
            password='Tharuicbt01',
            database='attendance_db'
        )
        if connection.is_connected():
            print("Connected to MySQL database")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def execute_query(query, values=None):
    """Executes a given SQL query (INSERT, UPDATE, DELETE)."""
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            if values:
                cursor.execute(query, values)
            else:
                cursor.execute(query)
            connection.commit()
            print("Query executed successfully")
        except Error as e:
            print(f"Error executing query: {e}")
        finally:
            cursor.close()
            connection.close()

def fetch_query(query, values=None):
    """Executes a SELECT query and returns the results."""
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            if values:
                cursor.execute(query, values)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Error as e:
            print(f"Error fetching data: {e}")
            return []
        finally:
            cursor.close()
            connection.close()

# Example usage:
if __name__ == "__main__":
    create_connection()  # Test database connection
