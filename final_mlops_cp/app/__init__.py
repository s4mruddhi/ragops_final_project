# app\__init__.py
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
