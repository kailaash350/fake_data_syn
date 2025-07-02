import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()


def setup_mysql_database():
    """Setup MySQL database for the fake data application"""

    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    database_name = os.getenv('DB_NAME', 'fake_data_db')

    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        print(f"Database '{database_name}' created successfully!")

        # Create user if needed (optional)
        # cursor.execute("CREATE USER IF NOT EXISTS 'fake_data_user'@'localhost' IDENTIFIED BY 'password123'")
        # cursor.execute(f"GRANT ALL PRIVILEGES ON {database_name}.* TO 'fake_data_user'@'localhost'")
        # cursor.execute("FLUSH PRIVILEGES")

        connection.commit()

    except mysql.connector.Error as e:
        print(f"Error setting up database: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


if __name__ == "__main__":
    setup_mysql_database()
