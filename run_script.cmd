@echo off

REM Path to the Python interpreter
set python_executable=python

REM Path to the Python file to be executed
set python_script=face_anonymize.py

REM Command-line arguments
set arguments=--image D:\face-blur\data --save_dir sav_res

REM Execute the Python file with arguments
%python_executable% %python_script% %arguments%

REM Pause the script
pause