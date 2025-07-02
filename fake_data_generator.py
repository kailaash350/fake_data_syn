import os
import pandas as pd
import mysql.connector
from faker import Faker
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import cProfile
import pstats
import io
from contextlib import contextmanager

# Load environment variables
load_dotenv()


class FakeDataGenerator:
    def __init__(self):
        self.fake = Faker(os.getenv('FAKER_LOCALE', 'en_US'))
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'fake_data_db')
        }
        self.records_count = int(os.getenv('RECORDS_COUNT', 1000))
        self.batch_size = int(os.getenv('BATCH_SIZE', 100))

        # Store loan IDs for referential integrity
        self.loan_ids = []

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

    def create_database_and_tables(self):
        """Create database and home loan related tables"""
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

                # Create home_loans table
                create_loans_table = """
                CREATE TABLE IF NOT EXISTS home_loans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    loan_id VARCHAR(50) UNIQUE NOT NULL,
                    borrower_first_name VARCHAR(100) NOT NULL,
                    borrower_last_name VARCHAR(100) NOT NULL,
                    borrower_email VARCHAR(150) NOT NULL,
                    borrower_phone VARCHAR(100),
                    borrower_ssn VARCHAR(11),
                    co_borrower_name VARCHAR(200),
                    loan_amount DECIMAL(15,2) NOT NULL,
                    interest_rate DECIMAL(5,2) NOT NULL,
                    loan_term_years INT NOT NULL,
                    monthly_payment DECIMAL(10,2) NOT NULL,
                    loan_type ENUM('Conventional', 'FHA', 'VA', 'USDA', 'Jumbo') NOT NULL,
                    loan_status ENUM('Active', 'Closed', 'Default', 'Pending', 'In Review') DEFAULT 'Active',
                    property_address TEXT NOT NULL,
                    property_city VARCHAR(100) NOT NULL,
                    property_state VARCHAR(50) NOT NULL,
                    property_zip VARCHAR(10) NOT NULL,
                    property_value DECIMAL(15,2) NOT NULL,
                    property_type ENUM('Single Family', 'Condo', 'Townhouse', 'Multi-Family', 'Mobile Home') NOT NULL,
                    down_payment DECIMAL(15,2) NOT NULL,
                    loan_to_value_ratio DECIMAL(5,2) NOT NULL,
                    debt_to_income_ratio DECIMAL(5,2),
                    credit_score INT NOT NULL,
                    annual_income DECIMAL(12,2) NOT NULL,
                    employment_status VARCHAR(50) NOT NULL,
                    loan_officer VARCHAR(100),
                    loan_start_date DATE NOT NULL,
                    loan_end_date DATE NOT NULL,
                    remaining_balance DECIMAL(15,2) NOT NULL,
                    escrow_balance DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_loan_id (loan_id),
                    INDEX idx_loan_status (loan_status),
                    INDEX idx_borrower_name (borrower_last_name, borrower_first_name),
                    INDEX idx_property_state (property_state)
                )
                """

                # Create loan_payments table
                create_payments_table = """
                CREATE TABLE IF NOT EXISTS loan_payments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    payment_id VARCHAR(50) UNIQUE NOT NULL,
                    loan_id VARCHAR(50) NOT NULL,
                    payment_date DATE NOT NULL,
                    due_date DATE NOT NULL,
                    scheduled_amount DECIMAL(10,2) NOT NULL,
                    actual_amount DECIMAL(10,2) NOT NULL,
                    principal_amount DECIMAL(10,2) NOT NULL,
                    interest_amount DECIMAL(10,2) NOT NULL,
                    escrow_amount DECIMAL(8,2) DEFAULT 0.00,
                    late_fee DECIMAL(8,2) DEFAULT 0.00,
                    payment_status ENUM('On Time', 'Late', 'Missed', 'Partial', 'Prepayment') NOT NULL,
                    days_late INT DEFAULT 0,
                    remaining_balance DECIMAL(15,2) NOT NULL,
                    payment_method ENUM('Auto Debit', 'Online', 'Check', 'Wire Transfer', 'Money Order') NOT NULL,
                    confirmation_number VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_payment_id (payment_id),
                    INDEX idx_loan_id (loan_id),
                    INDEX idx_payment_date (payment_date),
                    INDEX idx_payment_status (payment_status)
                )
                """

                # Create loan_modifications table
                create_modifications_table = """
                CREATE TABLE IF NOT EXISTS loan_modifications (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    modification_id VARCHAR(50) UNIQUE NOT NULL,
                    loan_id VARCHAR(50) NOT NULL,
                    modification_type ENUM('Rate Reduction', 'Term Extension', 'Principal Reduction', 'Payment Deferral', 'Forbearance') NOT NULL,
                    request_date DATE NOT NULL,
                    approval_date DATE,
                    effective_date DATE,
                    modification_status ENUM('Requested', 'Under Review', 'Approved', 'Denied', 'Active', 'Completed') NOT NULL,
                    old_interest_rate DECIMAL(5,2),
                    new_interest_rate DECIMAL(5,2),
                    old_monthly_payment DECIMAL(10,2),
                    new_monthly_payment DECIMAL(10,2),
                    old_term_years INT,
                    new_term_years INT,
                    reason_for_modification TEXT,
                    hardship_type ENUM('Job Loss', 'Income Reduction', 'Medical Emergency', 'Divorce', 'Death', 'Natural Disaster', 'Other'),
                    documentation_provided TEXT,
                    loan_officer VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_modification_id (modification_id),
                    INDEX idx_loan_id (loan_id),
                    INDEX idx_modification_status (modification_status)
                )
                """

                # Execute table creation
                cursor.execute(create_loans_table)
                cursor.execute(create_payments_table)
                cursor.execute(create_modifications_table)

                connection.commit()
                print("All home loan tables created successfully!")

        except mysql.connector.Error as e:
            print(f"Error creating database/tables: {e}")
            raise

    def generate_home_loan_record(self):
        """Generate a single home loan record"""
        loan_id = f"LOAN_{self.fake.unique.random_number(digits=10)}"

        # Borrower information
        first_name = self.fake.first_name()
        last_name = self.fake.last_name()

        # Property and loan details
        property_value = round(random.uniform(150000, 1500000), 2)
        down_payment_percent = random.uniform(0.03, 0.25)  # 3% to 25%
        down_payment = round(property_value * down_payment_percent, 2)
        loan_amount = property_value - down_payment

        interest_rate = round(random.uniform(2.5, 7.5), 2)
        loan_term_years = random.choice([15, 20, 25, 30])

        # Calculate monthly payment
        monthly_rate = interest_rate / 100 / 12
        num_payments = loan_term_years * 12
        if monthly_rate > 0:
            monthly_payment = round((loan_amount * monthly_rate * (1 + monthly_rate) ** num_payments) /
                                    ((1 + monthly_rate) ** num_payments - 1), 2)
        else:
            monthly_payment = round(loan_amount / num_payments, 2)

        # Loan dates
        loan_start_date = self.fake.date_between(start_date='-10y', end_date='today')
        loan_end_date = loan_start_date + timedelta(days=loan_term_years * 365)

        # Calculate remaining balance based on loan age
        months_elapsed = ((datetime.now().date() - loan_start_date).days // 30)
        remaining_balance = max(0, round(loan_amount * random.uniform(0.1, 0.95), 2))

        # Financial ratios
        ltv_ratio = round((loan_amount / property_value) * 100, 2)
        annual_income = round(random.uniform(40000, 300000), 2)
        monthly_income = annual_income / 12
        dti_ratio = round((monthly_payment / monthly_income) * 100, 2)

        record = {
            'loan_id': loan_id,
            'borrower_first_name': first_name,
            'borrower_last_name': last_name,
            'borrower_email': self.fake.email(),
            'borrower_phone': self.fake.phone_number(),
            'borrower_ssn': f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
            'co_borrower_name': self.fake.name() if random.choice([True, False]) else None,
            'loan_amount': loan_amount,
            'interest_rate': interest_rate,
            'loan_term_years': loan_term_years,
            'monthly_payment': monthly_payment,
            'loan_type': random.choice(['Conventional', 'FHA', 'VA', 'USDA', 'Jumbo']),
            'loan_status': random.choice(['Active', 'Active', 'Active', 'Closed', 'Default', 'Pending']),
            'property_address': self.fake.street_address(),
            'property_city': self.fake.city(),
            'property_state': self.fake.state(),
            'property_zip': self.fake.postcode(),
            'property_value': property_value,
            'property_type': random.choice(['Single Family', 'Condo', 'Townhouse', 'Multi-Family', 'Mobile Home']),
            'down_payment': down_payment,
            'loan_to_value_ratio': ltv_ratio,
            'debt_to_income_ratio': dti_ratio,
            'credit_score': random.randint(580, 850),
            'annual_income': annual_income,
            'employment_status': random.choice(['Employed', 'Self-Employed', 'Retired', 'Contract']),
            'loan_officer': self.fake.name(),
            'loan_start_date': loan_start_date,
            'loan_end_date': loan_end_date,
            'remaining_balance': remaining_balance,
            'escrow_balance': round(random.uniform(500, 5000), 2)
        }

        self.loan_ids.append(loan_id)
        return record

    def generate_payment_record(self):
        """Generate a single loan payment record"""
        if not self.loan_ids:
            return None

        loan_id = random.choice(self.loan_ids)

        # Payment dates
        due_date = self.fake.date_between(start_date='-2y', end_date='today')
        payment_date = due_date + timedelta(days=random.randint(-5, 30))  # Can be early or late

        scheduled_amount = round(random.uniform(800, 4500), 2)
        payment_status = random.choice(['On Time', 'On Time', 'On Time', 'Late', 'Missed', 'Partial', 'Prepayment'])

        # Calculate actual payment based on status
        if payment_status == 'Partial':
            actual_amount = round(scheduled_amount * random.uniform(0.3, 0.8), 2)
        elif payment_status == 'Missed':
            actual_amount = 0.00
        elif payment_status == 'Prepayment':
            actual_amount = round(scheduled_amount * random.uniform(1.1, 3.0), 2)
        else:
            actual_amount = scheduled_amount

        # Break down payment components
        interest_amount = round(actual_amount * random.uniform(0.3, 0.7), 2)
        escrow_amount = round(actual_amount * random.uniform(0.1, 0.3), 2)
        principal_amount = actual_amount - interest_amount - escrow_amount

        # Late fees and days late
        days_late = max(0, (payment_date - due_date).days) if payment_status in ['Late', 'Partial'] else 0
        late_fee = round(random.uniform(25, 150), 2) if days_late > 0 else 0.00

        record = {
            'payment_id': f"PAY_{self.fake.unique.random_number(digits=12)}",
            'loan_id': loan_id,
            'payment_date': payment_date,
            'due_date': due_date,
            'scheduled_amount': scheduled_amount,
            'actual_amount': actual_amount,
            'principal_amount': principal_amount,
            'interest_amount': interest_amount,
            'escrow_amount': escrow_amount,
            'late_fee': late_fee,
            'payment_status': payment_status,
            'days_late': days_late,
            'remaining_balance': round(random.uniform(50000, 500000), 2),
            'payment_method': random.choice(['Auto Debit', 'Online', 'Check', 'Wire Transfer', 'Money Order']),
            'confirmation_number': f"CONF_{self.fake.random_number(digits=8)}"
        }

        return record

    def generate_modification_record(self):
        """Generate a single loan modification record"""
        if not self.loan_ids:
            return None

        loan_id = random.choice(self.loan_ids)

        # Modification details
        modification_type = random.choice(
            ['Rate Reduction', 'Term Extension', 'Principal Reduction', 'Payment Deferral', 'Forbearance'])
        modification_status = random.choice(
            ['Requested', 'Under Review', 'Approved', 'Denied', 'Active', 'Completed'])

        # Dates
        request_date = self.fake.date_between(start_date='-3y', end_date='today')

        approval_date = None
        effective_date = None
        if modification_status in ['Approved', 'Active', 'Completed']:
            approval_date = request_date + timedelta(days=random.randint(15, 90))
            effective_date = approval_date + timedelta(days=random.randint(1, 30))

        # Old loan terms
        old_interest_rate = round(random.uniform(3.0, 8.0), 2)
        old_monthly_payment = round(random.uniform(1000, 4000), 2)
        old_term_years = random.choice([15, 20, 25, 30])

        # New loan terms (if approved)
        new_interest_rate = None
        new_monthly_payment = None
        new_term_years = None

        if modification_status in ['Approved', 'Active', 'Completed']:
            if modification_type == 'Rate Reduction':
                new_interest_rate = round(old_interest_rate - random.uniform(0.5, 2.0), 2)
                new_monthly_payment = round(old_monthly_payment * random.uniform(0.8, 0.95), 2)
                new_term_years = old_term_years
            elif modification_type == 'Term Extension':
                new_interest_rate = old_interest_rate
                new_term_years = old_term_years + random.choice([5, 10, 15])
                new_monthly_payment = round(old_monthly_payment * random.uniform(0.7, 0.9), 2)
            elif modification_type == 'Principal Reduction':
                new_interest_rate = old_interest_rate
                new_term_years = old_term_years
                new_monthly_payment = round(old_monthly_payment * random.uniform(0.6, 0.85), 2)
            else:  # Payment Deferral or Forbearance
                new_interest_rate = old_interest_rate
                new_term_years = old_term_years
                new_monthly_payment = round(old_monthly_payment * random.uniform(0.0, 0.5), 2)

        # Hardship and documentation
        hardship_types = ['Job Loss', 'Income Reduction', 'Medical Emergency', 'Divorce', 'Death',
                          'Natural Disaster', 'Other']
        hardship_type = random.choice(hardship_types)

        reasons = [
            "Financial hardship due to job loss",
            "Reduced income affecting payment ability",
            "Medical emergency causing financial strain",
            "Divorce settlement impacting finances",
            "Death of co-borrower",
            "Natural disaster property damage",
            "Economic downturn impact",
            "Temporary financial difficulty"
        ]

        documentation_options = [
            "Pay stubs, tax returns, hardship letter",
            "Medical bills, doctor's note, financial statements",
            "Unemployment benefits letter, bank statements",
            "Divorce decree, income documentation",
            "Death certificate, estate documents",
            "Insurance claims, repair estimates",
            "Financial statements, hardship affidavit"
        ]

        record = {
            'modification_id': f"MOD_{self.fake.unique.random_number(digits=10)}",
            'loan_id': loan_id,
            'modification_type': modification_type,
            'request_date': request_date,
            'approval_date': approval_date,
            'effective_date': effective_date,
            'modification_status': modification_status,
            'old_interest_rate': old_interest_rate,
            'new_interest_rate': new_interest_rate,
            'old_monthly_payment': old_monthly_payment,
            'new_monthly_payment': new_monthly_payment,
            'old_term_years': old_term_years,
            'new_term_years': new_term_years,
            'reason_for_modification': random.choice(reasons),
            'hardship_type': hardship_type,
            'documentation_provided': random.choice(documentation_options),
            'loan_officer': self.fake.name()
        }

        return record

    def generate_data_for_table(self, table_type):
        """Generate fake data for specific table type"""
        print(f"Generating {self.records_count} {table_type} records...")

        data = []
        for i in range(self.records_count):
            if i % 100 == 0:
                print(f"Generated {i} {table_type} records...")

            try:
                if table_type == 'home_loans':
                    record = self.generate_home_loan_record()
                elif table_type == 'loan_payments':
                    record = self.generate_payment_record()
                elif table_type == 'loan_modifications':
                    record = self.generate_modification_record()
                else:
                    print(f"Unknown table type: {table_type}")
                    continue

                if record:
                    data.append(record)

            except Exception as e:
                print(f"Error generating {table_type} record {i}: {e}")
                continue

        df = pd.DataFrame(data)
        print(f"Successfully generated {len(df)} {table_type} records")
        return df

    def load_data_to_mysql(self, df, table_name):
        """Load DataFrame to MySQL database"""
        try:
            # Create SQLAlchemy engine
            connection_string = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            engine = create_engine(connection_string)

            print(f"Loading {len(df)} records to {table_name} table...")

            # Load data in batches
            total_batches = (len(df) + self.batch_size - 1) // self.batch_size

            for i in range(0, len(df), self.batch_size):
                batch_num = (i // self.batch_size) + 1
                batch_df = df.iloc[i:i + self.batch_size]

                batch_df.to_sql(
                    name=table_name,
                    con=engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )

                print(f"Loaded batch {batch_num}/{total_batches} ({len(batch_df)} records) to {table_name}")

            print(f"Successfully loaded all {len(df)} records to {table_name}")

        except Exception as e:
            print(f"Error loading data to {table_name}: {e}")
            raise

    def verify_data_load(self):
        """Verify that data was loaded correctly"""
        try:
            with self.get_db_connection() as connection:
                cursor = connection.cursor()

                tables = ['home_loans', 'loan_payments', 'loan_modifications']

                for table in tables:
                    # Count total records
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"Total records in {table}: {count}")

                    # Show sample records
                    if count > 0:
                        cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                        columns = [desc[0] for desc in cursor.description]
                        sample_data = cursor.fetchall()

                        print(f"\nSample records from {table}:")
                        for row in sample_data:
                            record_dict = dict(zip(columns, row))
                            if table == 'home_loans':
                                print(
                                    f"  Loan ID: {record_dict['loan_id']}, Borrower: {record_dict['borrower_first_name']} {record_dict['borrower_last_name']}, Amount: ${record_dict['loan_amount']:,.2f}")
                            elif table == 'loan_payments':
                                print(
                                    f"  Payment ID: {record_dict['payment_id']}, Loan ID: {record_dict['loan_id']}, Amount: ${record_dict['actual_amount']:,.2f}, Status: {record_dict['payment_status']}")
                            elif table == 'loan_modifications':
                                print(
                                    f"  Modification ID: {record_dict['modification_id']}, Loan ID: {record_dict['loan_id']}, Type: {record_dict['modification_type']}, Status: {record_dict['modification_status']}")
                        print()

        except Exception as e:
            print(f"Error verifying data: {e}")

    def run_with_profiling(self):
        """Run the complete process with cProfile"""
        print("Starting home loan fake data generation and loading process...")

        # Create database and tables
        self.create_database_and_tables()

        # Generate and load data for each table
        tables = ['home_loans', 'loan_payments', 'loan_modifications']

        for table in tables:
            print(f"\n{'=' * 50}")
            print(f"Processing {table.upper()}")
            print(f"{'=' * 50}")

            # Generate fake data
            df = self.generate_data_for_table(table)

            if not df.empty:
                # Load data to MySQL
                self.load_data_to_mysql(df, table)
            else:
                print(f"No data generated for {table}")

        # Verify data load
        print(f"\n{'=' * 50}")
        print("VERIFICATION RESULTS")
        print(f"{'=' * 50}")
        self.verify_data_load()

        print("Process completed successfully!")

def profile_application():
    """Profile the application performance"""
    profiler = cProfile.Profile()

    try:
        generator = FakeDataGenerator()

        print("Starting profiled execution...")
        profiler.enable()

        generator.run_with_profiling()

        profiler.disable()

        # Print profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Show top 20 functions

        print("\n" + "=" * 50)
        print("PROFILING RESULTS")
        print("=" * 50)
        print(s.getvalue())

        # Save profiling results to file
        with open('profiling_results.txt', 'w') as f:
            f.write(s.getvalue())
        print("Profiling results saved to 'profiling_results.txt'")

    except Exception as e:
        profiler.disable()
        print(f"Error during profiled execution: {e}")
        raise

if __name__ == "__main__":
    profile_application()

