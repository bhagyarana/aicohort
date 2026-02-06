import asyncio
import logging
import os
import sqlite3
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# --- Logging Setup ---
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "mcp_server_activity.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode="w"),
    ],
)
# --- End Logging Setup ---

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database.db")

mcp_server = FastMCP(
       "sqlite-db-fast-mcp-server",    
    )


# --- Database Utility Functions ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # To access columns by name
    return conn


@mcp_server.tool
def list_db_tables(dummy_param: str) -> Dict[str, Any]:
    """Lists all tables in the SQLite database.

    Args:
        dummy_param (str): This parameter is not used by the function
                           but helps ensure schema generation. A non-empty string is expected.
    Returns:
        A dictionary with keys 'success' (bool), 'message' (str),
        and 'tables' (list[str]) containing the table names if successful.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return {
            "success": True,
            "message": "Tables listed successfully.",
            "tables": tables,
        }
    except sqlite3.Error as e:
        logging.error(f"Error listing tables: {e}")
        return {"success": False, "message": f"Error listing tables: {e}", "tables": []}


@mcp_server.tool
def get_table_schema(table_name: str) -> Dict[str, Any]:
    """Gets the schema (column names and types) of a specific table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        schema_info = cursor.fetchall()
        conn.close()
        if not schema_info:
            raise ValueError(f"Table '{table_name}' not found or no schema information.")

        columns = [{"name": row["name"], "type": row["type"]} for row in schema_info]
        return {"table_name": table_name, "columns": columns}
    except (sqlite3.Error, ValueError) as e:
        logging.error(f"Error getting schema for table '{table_name}': {e}")
        return {"success": False, "message": str(e)}


@mcp_server.tool
def query_db_table(table_name: str, columns: str, condition: str) -> List[Dict[str, Any]]:
    """Queries a table with an optional condition.

    Args:
        table_name: The name of the table to query.
        columns: Comma-separated list of columns to retrieve (e.g., "id, name"). Defaults to "*".
        condition: Optional SQL WHERE clause condition (e.g., "id = 1" or "completed = 0").
    Returns:
        A list of dictionaries, where each dictionary represents a row.
    """
    conn = get_db_connection()
    query = f"SELECT {columns} FROM {table_name}"
    if condition:
        query += f" WHERE {condition}"
    query += ";"

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        return results
    except sqlite3.Error as e:
        logging.error(f"Error querying table '{table_name}': {e}")
        # Return an error structure consistent with other tools
        return [{"success": False, "message": f"Error querying table '{table_name}': {e}"}]
    finally:
        conn.close()


@mcp_server.tool
def insert_data(table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Inserts a new row of data into the specified table.

    Args:
        table_name: The name of the table to insert data into.
        data: A dictionary where keys are column names and values are the
              corresponding values for the new row.
    Returns:
        A dictionary with keys 'success' (bool) and 'message' (str).
    """
    if not data:
        return {"success": False, "message": "No data provided for insertion."}

    conn = get_db_connection()
    cursor = conn.cursor()
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?" for _ in data])
    values = tuple(data.values())
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    try:
        cursor.execute(query, values)
        conn.commit()
        last_row_id = cursor.lastrowid
        return {
            "success": True,
            "message": f"Data inserted successfully. Row ID: {last_row_id}",
            "row_id": last_row_id,
        }
    except sqlite3.Error as e:
        conn.rollback()
        logging.error(f"Error inserting data into table '{table_name}': {e}")
        return {
            "success": False,
            "message": f"Error inserting data into table '{table_name}': {e}",
        }
    finally:
        conn.close()


@mcp_server.tool
def delete_data(table_name: str, condition: str) -> Dict[str, Any]:
    """Deletes rows from a table based on a given SQL WHERE clause condition.

    Args:
        table_name: The name of the table to delete data from.
        condition: The SQL WHERE clause condition to specify which rows to delete.
                   This condition MUST NOT be empty to prevent accidental mass deletion.
    Returns:
        A dictionary with keys 'success' (bool) and 'message' (str).
    """
    if not condition or not condition.strip():
        return {
            "success": False,
            "message": "Deletion condition cannot be empty. This is a safety measure.",
        }

    conn = get_db_connection()
    cursor = conn.cursor()
    query = f"DELETE FROM {table_name} WHERE {condition}"

    try:
        cursor.execute(query)
        rows_deleted = cursor.rowcount
        conn.commit()
        return {
            "success": True,
            "message": f"{rows_deleted} row(s) deleted successfully from table '{table_name}'.",
            "rows_deleted": rows_deleted,
        }
    except sqlite3.Error as e:
        conn.rollback()
        logging.error(f"Error deleting data from table '{table_name}': {e}")
        return {
            "success": False,
            "message": f"Error deleting data from table '{table_name}': {e}",
        }
    finally:
        conn.close()


# --- MCP Server Setup and Runner ---
if __name__ == "__main__":
    logging.info("Creating FastMCP Server instance for SQLite DB...")
    logging.info("Launching SQLite DB FastMCP Server via stdio...")
    try:
        # Run the server using the simplified stdio runner
        mcp_server.run(transport="stdio")
    except KeyboardInterrupt:
        logging.info("\nFastMCP Server (stdio) stopped by user.")
    except Exception as e:
        logging.critical(
            f"FastMCP Server (stdio) encountered an unhandled error: {e}", exc_info=True
        )
    finally:
        logging.info("FastMCP Server (stdio) process exiting.")