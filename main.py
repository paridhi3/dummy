#!/usr/bin/env python
import ast
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from io import StringIO
import traceback
import warnings
from typing import List, Dict, Any
from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
import pandas as pd
import cx_Oracle
from crewai import Crew, Process
from fastapi.responses import JSONResponse
from gmigrate_streamlit.tools.caching_tools import get_cached_response, get_prompt_hash, save_response_to_cache
from gmigrate_streamlit.tools.connect_db import connect_oracle, connect_snowflake
from gmigrate_streamlit.tools.snowflake_ddl_loader import load_ddl_snowflake
from gmigrate_streamlit.crew import GmigrateStreamlit
from gmigrate_streamlit.tools.oracle_ddl_extraction import (
    getOracleTableDDL,
    getOracleViewDDL,
    getOracleFunctionDDL,
    getOracleProcedureDDL,
)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

app = FastAPI(title="Agentic G-Migrate API")

# Pydantic models for request validation
class OracleCredentials(BaseModel):
    oracle_username: str
    oracle_password: str
    oracle_dsn: str

class SnowflakeCredentials(BaseModel):
    snowflake_username: str
    snowflake_password: str
    snowflake_account_identifier: str
    snowflake_warehouse: str
    snowflake_role: str

class GetObjectsRequest(BaseModel):
    schema: str
    object_type: str
    oracle_creds: OracleCredentials

class ExtractDDLRequest(BaseModel):
    schema: str
    objects: List[str]
    object_type: str
    oracle_creds: OracleCredentials

class AnalyzeRequest(BaseModel):
    source: str
    target: str
    object_type: str
    ddl: List[str]
    schema: str
    prog_language: str
    target_creds: SnowflakeCredentials

class MigrateRequest(BaseModel):
    source: str
    target: str
    object_type: str
    ddl: List[str]
    schema: str
    objects: List[str]
    prog_language: str
    target_creds: SnowflakeCredentials

# ------------------------ Helper functions ------------------------
curr_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")

def save_ddl_to_file(object_type: str, objects: List[str], ddl: List[str], curr_time: str):
    folder_name = "converted_ddls_folder"
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    file_map = {
        "TABLE": "table_ddl_file.txt",
        "VIEW": "view_ddl_file.txt",
        "FUNCTION": "function_ddl_file.txt",
        "PROCEDURE": "procedure_ddl_file.txt",
    }

    file_path = os.path.join(folder_path, file_map.get(object_type, "default_ddl_file.txt"))

    for i, converted_ddl in enumerate(ddl):
        with open(file_path, "a") as f:
            f.write("--------------------------------------------\n")
            f.write(f"{object_type.capitalize()} name: {objects[i]}\n")
            f.write(f"Time of Execution: {curr_time}\n")
            f.write(f"{converted_ddl}\n")

async def run_migrator(source: str, target: str, object_type: str, ddl: List[str], schema: str, snowflake_credentials: Dict[str, str], objects: List[str], prog_language: str):
    user_prompt = f"{source}{target}{object_type}{''.join(ddl)}{schema}Migrate{prog_language}"
    user_prompt_hash = get_prompt_hash(user_prompt)
    cached = get_cached_response(user_prompt_hash)

 
    if cached:
        migrator_result = cached
    else:
        inputs = {
            "source": source,
            "target": target,
            "object": object_type,
            "ddl": ddl,
            "schema": schema,
            "curr_time": curr_time,
            "prog_language": prog_language,
        }
        try:
            crew_instance = GmigrateStreamlit()
            migrator_agent = crew_instance.ddl_migrator()
            migrator_task = crew_instance.ddl_migrator_task()
            migrator_crew = Crew(
                agents=[migrator_agent],
                tasks=[migrator_task],
                process=Process.sequential,
                verbose=True,
            )
            migrator_result = migrator_crew.kickoff(inputs=inputs)
            migrator_result = str(migrator_result)
            save_response_to_cache(user_prompt, migrator_result, user_prompt_hash)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error running migrator: {str(e)}")

    try:
        migrator_result = ast.literal_eval(migrator_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing migrator result: {str(e)}")

    save_ddl_to_file(object_type, objects, migrator_result, curr_time)

    migrator_report = load_ddl_snowflake(schema, migrator_result, snowflake_credentials, objects)
    num_loaded = (migrator_report["Status"] == "Loaded").sum()
    num_not_loaded = (migrator_report["Status"] == "Not Loaded").sum()

    return {
        "csv_data": migrator_report.to_csv(index=False),
        "summary": {"Loaded": int(num_loaded), "Not Loaded": int(num_not_loaded)},
        "report": migrator_report.to_dict(orient="records"),
    }

async def run_analyzer(source: str, target: str, object_type: str, ddl: List[str], schema: str, prog_language: str):
    user_prompt = f"{source}{target}{object_type}{''.join(ddl)}{schema}Analyze{prog_language}"
    user_prompt_hash = get_prompt_hash(user_prompt)
    cached = get_cached_response(user_prompt_hash)

    if cached:
        analyzer_result = str(cached)
        df = pd.read_csv(StringIO(analyzer_result))
    else:
        inputs = {
            "source": source,
            "target": target,
            "object": object_type,
            "ddl": ddl,
            "schema": schema,
            "curr_time": curr_time,
            "prog_language": prog_language,
        }
        try:
            crew_instance = GmigrateStreamlit()
            analyzer_agent = crew_instance.ddl_analyzer()
            analyzer_task = crew_instance.ddl_analyzer_task()
            analyzer_crew = Crew(
                agents=[analyzer_agent],
                tasks=[analyzer_task],
                process=Process.sequential,
                verbose=True,
            )
            analyzer_result = analyzer_crew.kickoff(inputs=inputs)
            analyzer_result = str(analyzer_result)
            save_response_to_cache(user_prompt, analyzer_result, user_prompt_hash)
            df = pd.read_csv(StringIO(analyzer_result))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error running analyzer: {str(e)}")

    num_convertible = (df["Status"] == "convertible").sum()
    num_not_convertible = (df["Status"] == "not directly convertible").sum()

    return {
        "csv_data": df.to_csv(index=False),
        "summary": {"Convertible": int(num_convertible), "Not Directly Convertible": int(num_not_convertible)},
        "report": df.to_dict(orient="records"),
    }

# ------------------------ Endpoints ------------------------
@app.post("/test-oracle")
async def test_oracle(
    oracle_username: str = Form(...),
    oracle_password: str = Form(...),
    oracle_dsn: str = Form(...)
):
    try:
        connection = cx_Oracle.connect(
            user=oracle_username,
            password=oracle_password,
            dsn=oracle_dsn,
        )
        connection.close()
        return {"success": True, "message": "Oracle Connection Successful"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Oracle Connection failed: {str(e)}")

@app.post("/test-snowflake")
async def test_snowflake(
    snowflake_username: str = Form(...),
    snowflake_password: str = Form(...),
    snowflake_account: str = Form(...),
    snowflake_warehouse: str = Form(...),
    snowflake_database: str = Form(...),
    snowflake_schema: str = Form(...),
    snowflake_role: str = Form(...)
):
    try:
        creds = {
            "snowflake_username": snowflake_username,
            "snowflake_password": snowflake_password,
            "snowflake_account_identifier": snowflake_account,
            "snowflake_warehouse": snowflake_warehouse,
            "snowflake_database": snowflake_database,
            "snowflake_schema": snowflake_schema,
            "snowflake_role": snowflake_role
        }
        conn = connect_snowflake(creds)
        curr = conn.cursor()
        curr.execute("SELECT CURRENT_WAREHOUSE()")
        curr.fetchall()
        curr.execute("SELECT CURRENT_VERSION()")
        curr.fetchone()
        conn.close()
        return {"success": True, "message": "Snowflake Connection Successful"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Snowflake Connection failed: {str(e)}")
    
@app.post("/get-objects")
async def get_objects(
    oracle_username: str = Form(...),
    oracle_password: str = Form(...),
    oracle_dsn: str = Form(...),
    schema: str = Form(...),
    object_type: str = Form(...)
):
    try:
        connection = cx_Oracle.connect(
            user=oracle_username,
            password=oracle_password,
            dsn=oracle_dsn,
        )
        cursor = connection.cursor()

        if object_type == "SCHEMA(Tables only)" or object_type == "TABLE":
            cursor.execute(f"SELECT table_name FROM all_tables WHERE owner = '{schema}' ORDER BY table_name")
            objects = [row[0] for row in cursor.fetchall()]
        elif object_type == "VIEW":
            cursor.execute(f"SELECT view_name FROM all_views WHERE owner = '{schema}' ORDER BY view_name")
            objects = [row[0] for row in cursor.fetchall()]
        elif object_type == "FUNCTION":
            cursor.execute(
                f"SELECT object_name FROM all_objects WHERE owner = '{schema}' AND object_type = 'FUNCTION' ORDER BY object_name"
            )
            objects = [row[0] for row in cursor.fetchall()]
        elif object_type == "PROCEDURE":
            cursor.execute(
                f"SELECT object_name FROM all_objects WHERE owner = '{schema}' AND object_type = 'PROCEDURE' ORDER BY object_name"
            )
            objects = [row[0] for row in cursor.fetchall()]
        else:
            raise HTTPException(status_code=400, detail="Invalid object type")

        connection.close()
        return {"objects": objects}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch objects: {str(e)}")

@app.post("/extract-ddl")
async def extract_ddl(
    oracle_username: str = Form(...),
    oracle_password: str = Form(...),
    oracle_dsn: str = Form(...),
    schema: str = Form(...),
    object_type: str = Form(...),
    objects_raw: str = Form(...)  # Accept comma-separated string
):
    try:
        # Convert comma-separated string to list
        objects = [obj.strip().upper() for obj in objects_raw.split(",")]
        schema = schema.upper()

        creds = {
            "oracle_username": oracle_username,
            "oracle_password": oracle_password,
            "oracle_dsn": oracle_dsn
        }

        if object_type == "SCHEMA(Tables only)" or object_type == "TABLE":
            ddl = getOracleTableDDL(schema, objects, creds)
        elif object_type == "VIEW":
            ddl = getOracleViewDDL(schema, objects, creds)
        elif object_type == "FUNCTION":
            ddl = getOracleFunctionDDL(schema, objects, creds)
        elif object_type == "PROCEDURE":
            ddl = getOracleProcedureDDL(schema, objects, creds)
        else:
            raise HTTPException(status_code=400, detail="Invalid object type")

        return {"ddl": ddl}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract DDL: {str(e)}")

@app.post("/analyze")
async def analyze(
    source: str = Form(...),
    target: str = Form(...),
    object_type: str = Form(...),
    ddl: str = Form(...),
    schema: str = Form(...),
    prog_language: str = Form(...)
):
    try:
        result = await run_analyzer(
            source=source,
            target=target,
            object_type=object_type,
            ddl=ddl,
            schema=schema,
            prog_language=prog_language,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyzer failed: {str(e)}")

@app.post("/migrate")
async def migrate(
    source: str = Form(...),
    target: str = Form(...),
    object_type: str = Form(...),
    ddl: str = Form(...),
    schema: str = Form(...),
    snowflake_user: str = Form(...),
    snowflake_password: str = Form(...),
    snowflake_account: str = Form(...),
    snowflake_warehouse: str = Form(...),
    snowflake_database: str = Form(...),
    snowflake_schema: str = Form(...),
    snowflake_role: str = Form(...),
    prog_language: str = Form(...),
    objects: List[str] = Form(...)
):
    try:
        snowflake_credentials = {
            "user": snowflake_user,
            "password": snowflake_password,
            "account": snowflake_account,
            "warehouse": snowflake_warehouse,
            "database": snowflake_database,
            "schema": snowflake_schema,
            "role": snowflake_role
        }

        result = await run_migrator(
            source=source,
            target=target,
            object_type=object_type,
            ddl=ddl,
            schema=schema,
            snowflake_credentials=snowflake_credentials,
            objects=objects,
            prog_language=prog_language,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migrator failed: {str(e)}")
