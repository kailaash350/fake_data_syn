import os
import pandas as pd
import mysql.connector
from faker import Faker
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv
from sqlalchemy import create_engine
from contextlib import contextmanager

# Load environment variables
load_dotenv()


class CriticalDataGenerator:
    def __init__(self):
        self.fake = Faker('en_US')
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': 'pd_feature'
        }
        self.table_name = 'critical_data'
        self.records_count = int(os.getenv('RECORDS_COUNT', 1000))
        self.batch_size = int(os.getenv('BATCH_SIZE', 100))
        # Percentage of records that should be duplicates (based on customer_id)
        self.duplicate_percentage = float(os.getenv('DUPLICATE_PERCENTAGE', 0.15))  # 15% duplicates by default

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        connection = None
        try:
            connection = mysql.connector.connect(**self.db_config)
            yield connection
        except mysql.connector.Error as e:
            print(f"Database connection error: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()

    def create_database_and_table(self):
        """Create database and critical_data table"""
        try:
            # Connect without specifying database to create it
            temp_config = self.db_config.copy()
            temp_config.pop('database', None)

            with mysql.connector.connect(**temp_config) as connection:
                cursor = connection.cursor()

                # Create database
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']}")
                print(f"Database '{self.db_config['database']}' created or already exists")

                # Use the database
                cursor.execute(f"USE {self.db_config['database']}")

                # Create critical_data table - REMOVED UNIQUE constraint from customer_id to allow duplicates
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    customer_id VARCHAR(50) NOT NULL,
                    full_name VARCHAR(200) NOT NULL,
                    credit_card_number VARCHAR(19) NOT NULL,
                    account_balance DECIMAL(15,2) NOT NULL,
                    last_transaction_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_customer_id (customer_id),
                    INDEX idx_last_transaction_date (last_transaction_date)
                )
                """

                cursor.execute(create_table_query)
                connection.commit()
                print(f"Table '{self.table_name}' created or already exists")

        except mysql.connector.Error as e:
            print(f"Error creating database/table: {e}")
            raise

    def generate_credit_card_number(self):
        """Generate a realistic credit card number (Luhn algorithm compliant)"""
        # Common credit card prefixes
        prefixes = {
            'Visa': ['4'],
            'Mastercard': ['51', '52', '53', '54', '55'],
            'American Express': ['34', '37'],
            'Discover': ['6011', '65']
        }

        card_type = random.choice(list(prefixes.keys()))
        prefix = random.choice(prefixes[card_type])

        # Generate the rest of the number
        if card_type == 'American Express':
            # Amex has 15 digits
            remaining_digits = 15 - len(prefix) - 1  # -1 for check digit
        else:
            # Others have 16 digits
            remaining_digits = 16 - len(prefix) - 1  # -1 for check digit

        # Generate random digits
        number = prefix + ''.join([str(random.randint(0, 9)) for _ in range(remaining_digits)])

        # Calculate Luhn check digit
        def luhn_checksum(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]

            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10

        check_digit = (10 - luhn_checksum(int(number))) % 10
        full_number = number + str(check_digit)

        # Format with spaces for readability
        if len(full_number) == 15:  # Amex
            return f"{full_number[:4]} {full_number[4:10]} {full_number[10:]}"
        else:  # 16 digit cards
            return f"{full_number[:4]} {full_number[4:8]} {full_number[8:12]} {full_number[12:]}"

    def generate_critical_record(self, customer_id=None):
        """Generate a single critical data record"""
        first_name = self.fake.first_name()
        last_name = self.fake.last_name()

        record = {
            'customer_id': customer_id if customer_id else f"{self.fake.random_number(digits=8)}",
            'full_name': f"{first_name} {last_name}",
            'credit_card_number': self.generate_credit_card_number(),
            'account_balance': round(random.uniform(-5000, 100000), 2),
            'last_transaction_date': self.fake.date_between(start_date='-1y', end_date='today')
        }

        return record

    def generate_critical_data(self):
        """Generate critical fake data with duplicates and return as DataFrame"""
        print(
            f"Generating {self.records_count} critical data records with {self.duplicate_percentage * 100:.1f}% duplicates...")

        data = []
        customer_ids_for_duplicates = []

        # Calculate how many records should be duplicates
        duplicate_count = int(self.records_count * self.duplicate_percentage)
        unique_count = self.records_count - duplicate_count

        print(f"Creating {unique_count} unique records and {duplicate_count} duplicate records...")

        # Generate unique records first
        for i in range(unique_count):
            if i % 100 == 0:
                print(f"Generated {i} unique records...")

            try:
                record = self.generate_critical_record()
                data.append(record)

                # Randomly select some customer_ids for creating duplicates later
                if random.random() < 0.3:  # 30% chance to be selected for duplication
                    customer_ids_for_duplicates.append(record['customer_id'])

            except Exception as e:
                print(f"Error generating unique record {i}: {e}")
                continue

        # Generate duplicate records
        print(f"Creating {duplicate_count} duplicate records...")
        for i in range(duplicate_count):
            if i % 100 == 0:
                print(f"Generated {i} duplicate records...")

            try:
                # Select a random customer_id from existing records to duplicate
                if customer_ids_for_duplicates:
                    duplicate_customer_id = random.choice(customer_ids_for_duplicates)
                else:
                    # Fallback: use customer_id from any existing record
                    duplicate_customer_id = random.choice(data)['customer_id']

                # Generate a new record with the same customer_id but different other data
                record = self.generate_critical_record(customer_id=duplicate_customer_id)
                data.append(record)

            except Exception as e:
                print(f"Error generating duplicate record {i}: {e}")
                continue

        # Shuffle the data to mix unique and duplicate records
        random.shuffle(data)

        df = pd.DataFrame(data)

        # Print statistics about duplicates
        duplicate_customer_count = df['customer_id'].duplicated().sum()
        unique_customer_count = df['customer_id'].nunique()

        print(f"Successfully generated {len(df)} total records")
        print(f"Unique customer_ids: {unique_customer_count}")
        print(f"Duplicate records: {duplicate_customer_count}")
        print(f"Actual duplicate percentage: {(duplicate_customer_count / len(df)) * 100:.1f}%")

        return df

    def load_data_to_mysql(self, df):
        """Load DataFrame to MySQL database"""
        try:
            # Create SQLAlchemy engine
            connection_string = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            engine = create_engine(connection_string)

            print(f"Loading {len(df)} records to MySQL database...")

            # Load data in batches
            total_batches = (len(df) + self.batch_size - 1) // self.batch_size

            for i in range(0, len(df), self.batch_size):
                batch_num = (i // self.batch_size) + 1
                batch_df = df.iloc[i:i + self.batch_size]

                batch_df.to_sql(
                    name=self.table_name,
                    con=engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )

                print(f"Loaded batch {batch_num}/{total_batches} ({len(batch_df)} records)")

            print(f"Successfully loaded all {len(df)} records to {self.table_name}")

        except Exception as e:
            print(f"Error loading data to MySQL: {e}")
            raise

    def verify_data_load(self):
        """Verify that data was loaded correctly"""
        try:
            with self.get_db_connection() as connection:
                cursor = connection.cursor()

                # Count total records
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                count = cursor.fetchone()[0]
                print(f"Total records in {self.table_name}: {count}")

                # Count unique customer_ids
                cursor.execute(f"SELECT COUNT(DISTINCT customer_id) FROM {self.table_name}")
                unique_customers = cursor.fetchone()[0]
                print(f"Unique customer_ids: {unique_customers}")

                # Count duplicate customer_ids
                cursor.execute(f"""
                    SELECT COUNT(*) FROM (
                        SELECT customer_id 
                        FROM {self.table_name} 
                        GROUP BY customer_id 
                        HAVING COUNT(*) > 1
                    ) as duplicates
                """)
                customers_with_duplicates = cursor.fetchone()[0]
                print(f"Customer_ids with duplicates: {customers_with_duplicates}")

                # Show sample records (mask sensitive data for display)
                cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 5")
                columns = [desc[0] for desc in cursor.description]
                sample_data = cursor.fetchall()

                print("\nSample records (credit card numbers masked for security):")
                for row in sample_data:
                    record_dict = dict(zip(columns, row))
                    # Mask credit card number for display
                    masked_cc = record_dict['credit_card_number'][:4] + " **** **** " + record_dict[
                                                                                            'credit_card_number'][-4:]
                    print(f"Customer ID: {record_dict['customer_id']}, "
                          f"Name: {record_dict['full_name']}, "
                          f"CC: {masked_cc}, "
                          f"Balance: ${record_dict['account_balance']:,.2f}")

                # Show examples of duplicate customer_ids
                cursor.execute(f"""
                    SELECT customer_id, COUNT(*) as count 
                    FROM {self.table_name} 
                    GROUP BY customer_id 
                    HAVING COUNT(*) > 1 
                    LIMIT 5
                """)
                duplicate_examples = cursor.fetchall()

                if duplicate_examples:
                    print("\nExamples of duplicate customer_ids:")
                    for customer_id, count in duplicate_examples:
                        print(f"Customer ID: {customer_id} appears {count} times")

        except Exception as e:
            print(f"Error verifying data: {e}")

    def run_generation(self):
        """Run the complete process"""
        print("Starting critical data generation process...")

        # Create database and table
        self.create_database_and_table()

        # Generate fake data
        df = self.generate_critical_data()

        # Load data to MySQL
        self.load_data_to_mysql(df)

        # Verify data load
        self.verify_data_load()

        print("Critical data generation completed successfully!")


if __name__ == "__main__":
    generator = CriticalDataGenerator()
    generator.run_generation()
