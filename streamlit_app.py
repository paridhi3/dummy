#!/usr/bin/env python
import ast
import os
import traceback
import sys
import warnings
from io import StringIO
from crewai import Crew, Process
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
import cx_Oracle

from gmigrate_streamlit.tools.caching_tools import get_cached_response, get_prompt_hash, save_response_to_cache
from gmigrate_streamlit.tools.connect_db import connect_oracle, connect_snowflake
from gmigrate_streamlit.tools.snowflake_ddl_loader import load_ddl_data_snowflake, load_ddl_snowflake
curr_time = datetime.now(ZoneInfo("Asia/Kolkata"))
curr_time = curr_time.strftime("%Y-%m-%d %H:%M:%S")

from gmigrate_streamlit.crew import GmigrateStreamlit
from gmigrate_streamlit.tools.oracle_ddl_extraction import getOracleFunctionDDL, getOracleProcedureDDL, getOracleTableDDL, getOracleViewDDL

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# def get_user_prompt(purpose):
#     return str(source)+str(target)+str(object_type)+str(ddl)+str(schema)+str(purpose)


def run_migrator(source,target,object_type,ddl,schema,snowflake_credentials,objects,prog_language):
    user_prompt = str(source)+str(target)+str(object_type)+str(ddl)+str(schema)+str('Migrate')+str(prog_language)
    user_prompt_hash = get_prompt_hash(user_prompt)
    cached = get_cached_response(user_prompt_hash)
    if cached:
        st.write("Using Cached Data:")
        migrator_result = cached
    else:
        inputs = {
            "source":source,
            "target":target,
            "object":object_type,
            "ddl":ddl,
            "schema":schema,
            "curr_time":curr_time,
            "prog_language":prog_language
        }

        crew_instance = GmigrateStreamlit()
        st.write("Running migrator agent")
        migrator_agent = crew_instance.ddl_migrator()
        migrator_agent_task = crew_instance.ddl_migrator_task()
        migrator_crew = Crew(
            agents=[migrator_agent],
            tasks=[migrator_agent_task],
            process = Process.sequential,
            verbose=True
        )
        migrator_result = migrator_crew.kickoff(inputs=inputs)
        migrator_result = str(migrator_result)
        save_response_to_cache(user_prompt,migrator_result,user_prompt_hash)
    print("migrator_result : ", migrator_result)
    migrator_result = ast.literal_eval(migrator_result)
    print("migrator result", migrator_result)


    folder_name = "converted_ddls_folder"
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    table_ddl_file = "table_ddl_file.txt"
    view_ddl_file = "view_ddl_file.txt"
    function_ddl_file = "function_ddl_file.txt"
    procedure_ddl_file = "procedure_ddl_file.txt"


    if object_type == "TABLE":
        table_file_path = os.path.join(current_dir, folder_name, table_ddl_file)
    elif object_type == "VIEW":
        view_file_path = os.path.join(current_dir, folder_name, view_ddl_file)
    elif object_type == "FUNCTION":
        function_file_path = os.path.join(current_dir, folder_name, function_ddl_file)
    elif object_type == "PROCEDURE":
        procedure_file_path = os.path.join(current_dir, folder_name, procedure_ddl_file)


    for converted_ddl in range(len(migrator_result)):
        print(migrator_result[converted_ddl])
        if object_type == "TABLE":
            with open(table_file_path, "a") as f:
                f.write("--------------------------------------------" + "\n")
                f.write("Table name : " + objects[converted_ddl]+ "\n")
                f.write("Time of Execution : "+ curr_time + "\n")
                f.write(migrator_result[converted_ddl]+"\n")
        elif object_type == "VIEW":
             with open(view_file_path, "a") as f:
                f.write("---------------------------------------------" + "\n")
                f.write("View name : " + objects[converted_ddl]+"\n")
                f.write("Time of Execution : "+curr_time+"\n")
                f.write(migrator_result[converted_ddl]+"\n")
        elif object_type =="FUNCTION":
             with open(function_file_path, "a") as f:
                f.write("---------------------------------------------" + "\n")
                f.write("Function name : "+objects[converted_ddl]+"\n")
                f.write("Time of Execution : "+curr_time+"\n")
                f.write(migrator_result[converted_ddl]+"\n")
        elif object_type == "PROCEDURE":
            with open(procedure_file_path, "a") as f:
                f.write("---------------------------------------------" + "\n")
                f.write("Procedure name : "+objects[converted_ddl]+"\n")
                f.write("Time of Execution : "+curr_time+"\n")
                f.write(migrator_result[converted_ddl]+"\n")

            

    if object_type == "TABLE" or object_type == "SCHEMA(Tables only)":
        migrator_report = load_ddl_data_snowflake(schema,migrator_result,snowflake_credentials,objects, oracle_credentials)
    else:
        migrator_report = load_ddl_snowflake(schema,migrator_result,snowflake_credentials,objects)
        
    st.write("MIGRATOR AGENT REPORT")
    num_loaded = (migrator_report["Status"] == "Loaded").sum()
    num_not_loaded = (migrator_report["Status"] == "Not Loaded").sum()

    st.write("Loaded:", num_loaded)
    st.write("Not Loaded:", num_not_loaded)
    st.write(migrator_report)
    migrator_report.to_csv(f"{source}_{target}_{object_type}_migrator_output.csv", index=False)
    migrator_csv_data = migrator_report.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Migrator Report",
        data=migrator_csv_data,
        file_name=f"{source}_{target}_{object_type}_migrator_output.csv",
        mime="text/csv"
    )
def run_analyzer(source,target,object_type,ddl,schema, prog_language):
    try:
        user_prompt = str(source)+str(target)+str(object_type)+str(ddl)+str(schema)+str('Analyze')+str(prog_language)
        user_prompt_hash = get_prompt_hash(user_prompt)
        cached = get_cached_response(user_prompt_hash)
        if cached:
            cached_result = str(cached)
            df = pd.read_csv(StringIO(cached_result))
            st.write("Using Cached Data:")
        else:
            """
            Running Analyzer Agent
            """
            inputs = {
                "source":source,
                "target":target,
                "object":object_type,
                "ddl":ddl,
                "schema":schema,
                "curr_time":curr_time,
                "prog_language":prog_language
            }
        
            try:
                crew_instance = GmigrateStreamlit()
                analyzer_agent = crew_instance.ddl_analyzer()
                analyzer_task = crew_instance.ddl_analyzer_task()

                analyzer_crew = Crew(
                    agents=[analyzer_agent],
                    tasks=[analyzer_task],
                    process = Process.sequential,
                    verbose=True
                )

                analyzer_result = analyzer_crew.kickoff(inputs=inputs)
                # result = GmigrateStreamlit().crew().kickoff(inputs=inputs)
                analyzer_result = str(analyzer_result)
                print('ANALYZER RESULT : ',analyzer_result)
                save_response_to_cache(user_prompt,analyzer_result,user_prompt_hash)
                df = pd.read_csv(StringIO(analyzer_result))
            except Exception as e:
                traceback.print_exc()
                raise Exception(f"An error occurred while running the crew: {e}")
        num_convertible = (df["Status"] == "convertible").sum()
        num_not_convertible = (df["Status"] == "not directly convertible").sum()
        st.write("ANALYZER AGENT REPORT")
        st.write("Convertible:", num_convertible)
        st.write("Not Directly Convertible:", num_not_convertible)
        st.write(df)
        df.to_csv(f"{source}_{object_type}_analyzer_output.csv", index=False)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Analyzer Report",
            data=csv_data,
            file_name=f"{source}_{object_type}_analyzer_output.csv",
            mime="text/csv"
        )
    except Exception as e:
        traceback.print_exc()
        raise Exception(f"An error occurred while running the crew: {e}")


# --------------------------------------------------STREAMLIT PAGE START-------------------------------------------------
st.title("Agentic G-Migrate")

if "source_button_clicked" not in st.session_state:
    st.session_state.source_button_clicked = False
if "target_button_clicked" not in st.session_state:
    st.session_state.target_button_clicked = False
if "analyze_button_clicked" not in st.session_state:
    st.session_state.analyze_button_clicked = False
if "migrate_button_clicked" not in st.session_state:
    st.session_state.migrate_button_clicked = False






col1, col2 = st.columns(2)

# ---------------- SOURCE DB (Oracle) ---------------- #
with col1:
    source = st.selectbox(
        "Select Source Database",
        ["ORACLE"]
    )

    if source == "ORACLE":
        oracle_username = st.text_input("Enter Oracle Username")
        oracle_password = st.text_input("Enter Oracle Password", type="password")
        oracle_dsn = st.text_input("Enter Oracle DSN")
        oracle_credentials = {"oracle_username":oracle_username,"oracle_password":oracle_password,"oracle_dsn":oracle_dsn}
        
        if st.button("Test Oracle Connection"):
            try:
                connection = cx_Oracle.connect(
                    user=oracle_username,
                    password=oracle_password,
                    dsn=oracle_dsn
                )
                st.session_state.source_button_clicked = True
                st.success("Oracle Connection Successful")
            except Exception as e:
                st.session_state.source_button_clicked = False
                st.error(f"Oracle Connection failed: {e}")

# ---------------- TARGET DB (Snowflake) ---------------- #
with col2:
    target = st.selectbox(
        "Select Target Database",
        ["SNOWFLAKE", "REDSHIFT", "DATABRICKS", "POSTGRE", "MYSQL"]
    )

    if target == "SNOWFLAKE":
        snowflake_username = st.text_input("Enter Snowflake Username")
        snowflake_password = st.text_input("Enter Snowflake Password", type="password")
        snowflake_account_identifier = st.text_input("Enter Snowflake Account Identifier")
        snowflake_warehouse = st.text_input("Enter Snowflake Warehouse Name")
        snowflake_role = st.text_input("Enter Snowflake Role")

        snowflake_credentials = {"snowflake_username":snowflake_username,"snowflake_password":snowflake_password,"snowflake_account_identifier":snowflake_account_identifier,"snowflake_warehouse":snowflake_warehouse,"snowflake_role":snowflake_role}

        if st.button("Test Snowflake Connection"):
            try:
                conn = connect_snowflake(snowflake_credentials)
                curr = conn.cursor()
                curr.execute("SELECT CURRENT_WAREHOUSE()")
                print(curr.fetchall())
                curr.execute("SELECT CURRENT_VERSION()")
                curr.fetchone()
                st.session_state.target_button_clicked = True
                st.success("Snowflake Connection successful")
            except Exception as e:
                st.session_state.target_button_clicked = False
                st.error(f"Snowflake connection failed: {e}")


if st.session_state.get("source_button_clicked") and st.session_state.get("target_button_clicked"):
    # Create 3 columns for spacing: left, center-left, center-right
    empty1, col_analyze, col_migrate, empty2 = st.columns([1, 1, 1, 1])

    with col_analyze:
        if st.button("ANALYZE", use_container_width=True):
            st.session_state.migrate_button_clicked = False
            st.session_state.analyze_button_clicked = True

    with col_migrate:
        if st.button("MIGRATE", use_container_width=True):
            st.session_state.analyze_button_clicked = False
            st.session_state.migrate_button_clicked = True


if (source == "ORACLE" and st.session_state.analyze_button_clicked == True) or (source == "ORACLE" and st.session_state.migrate_button_clicked == True):
    if(st.session_state.analyze_button_clicked == True):
        st.markdown("**Analyzer**")
    elif(st.session_state.migrate_button_clicked == True):
        st.markdown("**Migrator**")
    objects=[]
    ddl=[]
    schema = st.text_input(f"Enter {source} Schema Name").upper()

    if "object_type_dropdown" not in st.session_state:
        st.session_state.object_type_dropdown = False

    if "prog_language" not in st.session_state:
        st.session_state.prog_language=""
    st.session_state.object_type_dropdown = st.selectbox(
        "Select Database Object",
        ["SCHEMA(Tables only)","TABLE", "VIEW", "FUNCTION", "PROCEDURE"]
    )

    try:
        connection = connect_oracle(oracle_credentials)
        orclcur = connection.cursor()

        if st.session_state.object_type_dropdown == "SCHEMA(Tables only)":
            orclcur.execute(f"SELECT table_name FROM all_tables WHERE owner = '{schema}' ORDER BY table_name")
            tables = [row[0] for row in orclcur.fetchall()]
            objects_str = tables
            objects=objects_str
        elif st.session_state.object_type_dropdown == "TABLE":
            orclcur.execute(f"SELECT table_name FROM all_tables WHERE owner = '{schema}' ORDER BY table_name")
            tables = [row[0] for row in orclcur.fetchall()]
            all_tables = ["Select All"] + tables
            objects_str = st.multiselect(
                f"Select {st.session_state.object_type_dropdown} names",
                all_tables
            )
            if "Select All" in objects_str:
                objects_str = tables

            objects=objects_str
        elif st.session_state.object_type_dropdown == "VIEW":
            orclcur.execute(f"SELECT view_name FROM all_views WHERE owner = '{schema}' ORDER BY view_name")
            views = [row[0] for row in orclcur.fetchall()]
            all_views = ["Select All"] + views
            objects_str = st.multiselect(
                f"Select {st.session_state.object_type_dropdown} names",
                all_views
            )
            if "Select All" in objects_str:
                objects_str = views

            objects=objects_str
        elif st.session_state.object_type_dropdown == "FUNCTION":
            if st.session_state.migrate_button_clicked == True or st.session_state.analyze_button_clicked == True:      
                st.session_state.prog_language = st.selectbox(
                    "Select Programming Language for target function",
                    ["SQL","Javascript", "Python", "Java", "Scala"]
                )
            orclcur.execute(f"SELECT object_name AS function_name FROM all_objects WHERE owner = '{schema}' AND object_type = 'FUNCTION' ORDER BY object_name")
            functions = [row[0] for row in orclcur.fetchall()]
            all_functions = ["Select All"] + functions
            objects_str = st.multiselect(
                f"Select {st.session_state.object_type_dropdown} names",
                all_functions
            )
            if "Select All" in objects_str:
                objects_str = functions

            objects=objects_str
        elif st.session_state.object_type_dropdown == "PROCEDURE":
            if  st.session_state.migrate_button_clicked == True or st.session_state.analyze_button_clicked == True:
                st.session_state.prog_language = st.selectbox(
                    "Select Programming Language for target procedure",
                    ["Snowflake Scripting(SQL)","Javascript", "Python", "Java", "Scala"]
                )
            orclcur.execute(f"SELECT object_name AS procedure_name FROM all_objects WHERE owner = '{schema}' AND object_type = 'PROCEDURE' ORDER BY object_name")
            procedures = [row[0] for row in orclcur.fetchall()]
            all_procedures = ["Select All"] + procedures
            objects_str = st.multiselect(
                f"Select {st.session_state.object_type_dropdown} names",
                all_procedures
            )
            if "Select All" in objects_str:
                objects_str = procedures

            objects=objects_str
    except Exception as e:
        st.write(e)
    if st.button("Submit"):
        if st.session_state.object_type_dropdown == "SCHEMA(Tables only)":
            ddl = getOracleTableDDL(schema,objects,oracle_credentials)
        elif st.session_state.object_type_dropdown == "TABLE":
            ddl = getOracleTableDDL(schema, objects, oracle_credentials)
        elif st.session_state.object_type_dropdown == "VIEW":
            ddl = getOracleViewDDL(schema,objects,  oracle_credentials)
        elif st.session_state.object_type_dropdown == "FUNCTION":
            ddl = getOracleFunctionDDL(schema, objects, oracle_credentials)
        elif st.session_state.object_type_dropdown == "PROCEDURE":
            ddl = getOracleProcedureDDL(schema,objects, oracle_credentials)

        if st.session_state.analyze_button_clicked:
            run_analyzer(source,target, st.session_state.object_type_dropdown,ddl,schema,st.session_state.prog_language)
        elif st.session_state.migrate_button_clicked:
            run_migrator(source,target,st.session_state.object_type_dropdown,ddl,schema, snowflake_credentials,objects,st.session_state.prog_language)
            
