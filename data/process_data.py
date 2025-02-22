import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine

# database_filepath = "data/DisasterPipeline.db"
# messages_filepath = "data/disaster_messages.csv"
# categories_filepath = "data/disaster_categories.csv"


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge message and category data from CSV files.

    Parameters:
    messages_filepath (str): Filepath to the CSV file containing message data.
    categories_filepath (str): Filepath to the CSV file containing category data.

    Returns:
    df (DataFrame): Merged DataFrame containing both message and category data.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on = 'id')

    return df


def clean_data(df):
    """
    Clean the DataFrame containing message and category data.

    This function splits the 'categories' column into separate category columns,
    cleans up the category values, and merges them back into the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing message and category data.

    Returns:
    df (DataFrame): Cleaned DataFrame with separate category columns.
    """

    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    df.drop('categories', inplace=True, axis=1)
    df = pd.concat([df,categories], axis = 1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filepath):
    """
    Save DataFrame to a SQLite database.

    This function saves the DataFrame to the specified SQLite database file
    using SQLAlchemy's create_engine method. The table name in the database
    is set to 'disastertable'.

    Parameters:
    df (DataFrame): Input DataFrame to be saved.
    database_filepath (str): Filepath to the SQLite database.

    Returns:
    None
    """

    # Create a SQLAlchemy engine to connect to the SQLite database
    engine = create_engine(f"sqlite:///{database_filepath}")

    # Save the DataFrame to the database with the default table name (the name of the DataFrame)
    df.to_sql("disastertable", engine, if_exists="replace", index=False)
    
    print(f"Data successfully saved to {database_filepath}")

def main():
    """
    Process raw message and category data, clean it, and save it to a database.

    This function is the entry point for processing data. It takes three command-line arguments:
    - The filepath of the CSV file containing message data.
    - The filepath of the CSV file containing category data.
    - The filepath of the SQLite database to save the cleaned data to.

    It loads message and category data from the specified CSV files, cleans the data using the clean_data function,
    and saves the cleaned data to the specified database using the save_data function.

    Parameters:
    None

    Returns:
    None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()