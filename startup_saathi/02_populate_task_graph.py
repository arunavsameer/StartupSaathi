# Databricks notebook source
# ============================================================
# NOTEBOOK 02: Populate Task Graph (DAG Data)
# StartupSaathi — Legal Navigator | Bharat Bricks Hacks 2026
#
# PURPOSE: Writes all compliance tasks with their dependency
#          relationships into the task_graph Delta table.
# RUN: Once. Safe to re-run — it overwrites the table.
# ============================================================

# COMMAND ----------

# Quick sanity check — confirm catalog access
spark.sql("USE CATALOG startup_hackathon")
spark.sql("USE SCHEMA legal_data")
print("✅ Catalog and schema set.")

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, ArrayType
)

# ─────────────────────────────────────────────────────────────
# TASK DATA — 17 compliance tasks with full dependency graph.
# Each task has explicit prereq_ids so the app can run
# Kahn's Topological Sort to enforce the correct order.
#
# To ADD a new task later: just add a dict to this list
# and re-run this notebook. The app will pick it up automatically.
# ─────────────────────────────────────────────────────────────

TASKS = [
    {
        "task_id":       "T001",
        "task_name":     "Choose Business Structure",
        "description":   "Decide between Private Limited Company, LLP, OPC, or Sole Proprietorship. This choice affects all downstream registrations and compliance requirements.",
        "authority":     "Self / CA Advisor",
        "portal_url":    "https://www.mca.gov.in/mcafoportal/showdataDict.do",
        "prereq_ids":    [],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "incorporation",
        "est_days":      2,
    },
    {
        "task_id":       "T002",
        "task_name":     "Name Reservation via RUN (Reserve Unique Name)",
        "description":   "Reserve your company name through the MCA RUN service. The name must comply with the Companies Act 2013 naming guidelines and not conflict with existing trademarks.",
        "authority":     "MCA",
        "portal_url":    "https://www.mca.gov.in/mcafoportal/showDataByAppType.do?type=roc&appType=17",
        "prereq_ids":    ["T001"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "incorporation",
        "est_days":      3,
    },
    {
        "task_id":       "T003",
        "task_name":     "Register Company via MCA SPICe+ Form",
        "description":   "File the SPICe+ (Simplified Proforma for Incorporating Company Electronically Plus) form on the MCA portal. This single form handles PAN, TAN, GSTIN, EPFO, ESIC, and bank account in one go.",
        "authority":     "MCA",
        "portal_url":    "https://www.mca.gov.in/mcafoportal/showdataDict.do",
        "prereq_ids":    ["T002"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "incorporation",
        "est_days":      5,
    },
    {
        "task_id":       "T004",
        "task_name":     "Obtain Certificate of Incorporation (COI)",
        "description":   "Receive the Certificate of Incorporation from MCA confirming your company's legal existence. This is the foundation document required for all subsequent registrations.",
        "authority":     "MCA / ROC",
        "portal_url":    "https://www.mca.gov.in/mcafoportal/login.do",
        "prereq_ids":    ["T003"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "incorporation",
        "est_days":      7,
    },
    {
        "task_id":       "T005",
        "task_name":     "Apply for PAN & TAN",
        "description":   "Obtain the company's Permanent Account Number (PAN) for tax purposes and Tax Deduction Account Number (TAN) for TDS filings. Both are required before opening a bank account.",
        "authority":     "Income Tax Department / NSDL",
        "portal_url":    "https://www.tin-nsdl.com/",
        "prereq_ids":    ["T004"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "incorporation",
        "est_days":      3,
    },
    {
        "task_id":       "T006",
        "task_name":     "Open Current Bank Account",
        "description":   "Open a current account in the company's name using the COI, PAN, and MOA/AOA. Required for all financial transactions and GST registration.",
        "authority":     "Any Scheduled Bank",
        "portal_url":    "https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=20547",
        "prereq_ids":    ["T005"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "incorporation",
        "est_days":      3,
    },
    {
        "task_id":       "T007",
        "task_name":     "GST Registration",
        "description":   "Register for Goods and Services Tax if annual turnover exceeds ₹40 lakhs (₹20 lakhs for services). Mandatory for interstate supply regardless of turnover.",
        "authority":     "GSTN",
        "portal_url":    "https://www.gst.gov.in/",
        "prereq_ids":    ["T006"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "post-incorporation",
        "est_days":      7,
    },
    {
        "task_id":       "T008",
        "task_name":     "DPIIT Startup Recognition",
        "description":   "Apply for official startup recognition from DPIIT to access tax exemptions (Section 80IAC), self-certification of labor and environmental laws, and other Startup India benefits.",
        "authority":     "DPIIT / Startup India",
        "portal_url":    "https://www.startupindia.gov.in/content/sih/en/startupgov/startup-recognition-page.html",
        "prereq_ids":    ["T004"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "post-incorporation",
        "est_days":      10,
    },
    {
        "task_id":       "T009",
        "task_name":     "Shops & Establishment Registration",
        "description":   "Register under the state Shops and Establishments Act applicable to your state. Required for all commercial establishments employing workers. Rules vary by state.",
        "authority":     "State Labour Department",
        "portal_url":    "https://shramsuvidha.gov.in/",
        "prereq_ids":    ["T004"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "post-incorporation",
        "est_days":      7,
    },
    {
        "task_id":       "T010",
        "task_name":     "Professional Tax Registration",
        "description":   "Register for Professional Tax (PT) in applicable states. Required for employers to deduct PT from employee salaries. Applicable in Maharashtra, Karnataka, West Bengal, and several other states.",
        "authority":     "State Commercial Tax Department",
        "portal_url":    "https://www.mahagst.gov.in/en/professional-tax",
        "prereq_ids":    ["T007"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "post-incorporation",
        "est_days":      5,
    },
    {
        "task_id":       "T011",
        "task_name":     "ESIC Registration (if > 10 employees)",
        "description":   "Register with the Employees' State Insurance Corporation if you have 10 or more employees earning under ₹21,000/month. Provides medical and cash benefits to employees.",
        "authority":     "ESIC",
        "portal_url":    "https://www.esic.in/",
        "prereq_ids":    ["T006"],
        "sector_filter": ["all"],
        "size_filter":   ["small", "medium"],
        "phase":         "operations",
        "est_days":      7,
    },
    {
        "task_id":       "T012",
        "task_name":     "EPF Registration (if > 20 employees)",
        "description":   "Register with EPFO (Employees' Provident Fund Organisation) if you have 20 or more employees. Both employer (12% of basic salary) and employee contributions are mandatory.",
        "authority":     "EPFO",
        "portal_url":    "https://www.epfindia.gov.in/",
        "prereq_ids":    ["T006"],
        "sector_filter": ["all"],
        "size_filter":   ["medium"],
        "phase":         "operations",
        "est_days":      7,
    },
    {
        "task_id":       "T013",
        "task_name":     "Gratuity Policy Setup (if > 10 employees)",
        "description":   "Set up a gratuity policy compliant with the Payment of Gratuity Act 1972. Employees who complete 5+ years of service are entitled to gratuity. Consider a group gratuity insurance policy.",
        "authority":     "Labour Department / Insurance Provider",
        "portal_url":    "https://labour.gov.in/whatsnew/payment-gratuity-act-1972",
        "prereq_ids":    ["T011"],
        "sector_filter": ["all"],
        "size_filter":   ["small", "medium"],
        "phase":         "operations",
        "est_days":      10,
    },
    {
        "task_id":       "T014",
        "task_name":     "FSSAI License (Food Business Operators)",
        "description":   "Obtain FSSAI Basic Registration (turnover < ₹12L), State License (₹12L–20Cr), or Central License (> ₹20Cr or multi-state operations). Mandatory for any food-related business.",
        "authority":     "FSSAI",
        "portal_url":    "https://foscos.fssai.gov.in/",
        "prereq_ids":    ["T004"],
        "sector_filter": ["food_tech"],
        "size_filter":   ["all"],
        "phase":         "post-incorporation",
        "est_days":      30,
    },
    {
        "task_id":       "T015",
        "task_name":     "Factory License under Factories Act 1948",
        "description":   "Obtain a Factory License from the state Chief Inspector of Factories before commencing manufacturing operations. Required if the premises employs 10+ workers with power or 20+ without power.",
        "authority":     "State Factory Inspectorate",
        "portal_url":    "https://shramsuvidha.gov.in/",
        "prereq_ids":    ["T004"],
        "sector_filter": ["manufacturing"],
        "size_filter":   ["all"],
        "phase":         "post-incorporation",
        "est_days":      30,
    },
    {
        "task_id":       "T016",
        "task_name":     "Contract Labour Registration (Principal Employer)",
        "description":   "Register as a Principal Employer under the Contract Labour (Regulation and Abolition) Act 1970 if you engage 20+ contract workers. Also ensure each contractor obtains a licence.",
        "authority":     "State Labour Department",
        "portal_url":    "https://shramsuvidha.gov.in/",
        "prereq_ids":    ["T015"],
        "sector_filter": ["manufacturing"],
        "size_filter":   ["small", "medium"],
        "phase":         "operations",
        "est_days":      15,
    },
    {
        "task_id":       "T017",
        "task_name":     "Trademark Filing (Strongly Recommended)",
        "description":   "File a trademark application for your brand name and logo under the Trade Marks Act 1999. TM status is granted immediately; registration typically takes 18–24 months. Early filing protects priority.",
        "authority":     "Office of the Controller General of Patents, Designs & Trade Marks",
        "portal_url":    "https://ipindia.gov.in/trade-marks.htm",
        "prereq_ids":    ["T004"],
        "sector_filter": ["all"],
        "size_filter":   ["all"],
        "phase":         "post-incorporation",
        "est_days":      5,
    },
]

print(f"✅ Loaded {len(TASKS)} tasks.")

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, ArrayType
)

# Define explicit schema so Spark doesn't infer incorrectly
schema = StructType([
    StructField("task_id",       StringType(),              False),
    StructField("task_name",     StringType(),              True),
    StructField("description",   StringType(),              True),
    StructField("authority",     StringType(),              True),
    StructField("portal_url",    StringType(),              True),
    StructField("prereq_ids",    ArrayType(StringType()),   True),
    StructField("sector_filter", ArrayType(StringType()),   True),
    StructField("size_filter",   ArrayType(StringType()),   True),
    StructField("phase",         StringType(),              True),
    StructField("est_days",      IntegerType(),             True),
])

# Convert list-of-dicts to Spark DataFrame using the explicit schema
from pyspark.sql import Row

rows = [Row(**task) for task in TASKS]
df = spark.createDataFrame(rows, schema=schema)

print(f"✅ DataFrame created. Row count: {df.count()}")
df.printSchema()
df.show(5, truncate=50)

# COMMAND ----------

# Write to Delta table (overwrite — safe to re-run)
(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("startup_hackathon.legal_data.task_graph")
)

print("✅ task_graph written to Delta table.")

# COMMAND ----------

# Verify the write
count = spark.sql("SELECT COUNT(*) AS total FROM startup_hackathon.legal_data.task_graph").collect()[0]["total"]
print(f"✅ Rows in task_graph: {count}")

spark.sql("""
    SELECT task_id, task_name, phase, SIZE(prereq_ids) AS num_prereqs, sector_filter
    FROM startup_hackathon.legal_data.task_graph
    ORDER BY phase, task_id
""").show(20, truncate=40)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# DEPENDENCY GRAPH VALIDATION
# Make sure no task references a prereq_id that doesn't exist.
# ─────────────────────────────────────────────────────────────

all_task_ids = {t["task_id"] for t in TASKS}
errors = []
for task in TASKS:
    for prereq in task["prereq_ids"]:
        if prereq not in all_task_ids:
            errors.append(f"Task {task['task_id']} references unknown prereq: {prereq}")

if errors:
    for e in errors:
        print(f"❌ {e}")
    raise ValueError("Dependency graph has broken references. Fix before proceeding.")
else:
    print("✅ All dependency references are valid.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# TOPOLOGICAL SORT TEST (Kahn's Algorithm)
# Validates that the task graph has no cycles.
# ─────────────────────────────────────────────────────────────

from collections import defaultdict, deque

graph = defaultdict(list)
in_degree = {t["task_id"]: 0 for t in TASKS}

for task in TASKS:
    for prereq in task["prereq_ids"]:
        graph[prereq].append(task["task_id"])
        in_degree[task["task_id"]] += 1

queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
topo_order = []

while queue:
    node = queue.popleft()
    topo_order.append(node)
    for neighbor in graph[node]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)

if len(topo_order) != len(TASKS):
    raise ValueError(f"❌ Cycle detected in task graph! Only processed {len(topo_order)}/{len(TASKS)} tasks.")
else:
    print(f"✅ Topological sort successful. Order: {' → '.join(topo_order)}")