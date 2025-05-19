
# 🔄 GenAI-Powered Source to Target Data Mapping

## 📘 Overview

This project leverages Generative AI to automate the traditionally manual and error-prone task of source-to-target data mapping. By combining robust data preprocessing techniques, prompt-driven transformation logic, and an intuitive Streamlit-based user interface, the system transforms raw source tables into structured target formats. The project supports querying, visualization, and dynamic data transformations using language models like GPT-4.

---

## 🔍 What the Project Does

- Preprocesses raw CSV files with dynamic data type detection, missing value handling, and categorical encoding.
- Uses a prompt-based LLM (GPT-4) to analyze source and sample tables, infer column mappings, and generate transformation code.
- Executes the generated code to produce a transformed **target.csv** file.
- Displays results, mappings, and transformation logic through a friendly **Streamlit UI**.
- Visualizes column mappings with a **network graph** using Plotly.
- Allows querying of the generated target table using natural language interpreted via GPT.

---

## 💡 Why the Project Is Useful

Traditional data mapping is manual, time-consuming, and prone to human error. This solution automates the process, improving:
- 🔍 **Accuracy** — Data type-aware transformations and smart column detection.
- ⚡ **Speed** — Auto-generated transformation logic saves developer time.
- 🤖 **Scalability** — Handles large CSVs without merge/concat overhead.
- 🎯 **Accessibility** — Designed for both technical and non-technical users.

---

## 🚀 How to Get Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/genai-data-mapping.git
cd genai-data-mapping
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

> Includes: `pandas`, `scikit-learn`, `streamlit`, `plotly`, `networkx`, `openai`, `langchain`, `dateutil`

### 3. Prepare Your Files

- Place your **source table** as `source.csv`
- Place your **sample target structure** as `sample_data.csv`
- Update your **OpenAI API key** in `test.py` where `OpenAI_API_KEY = 'Input Key'`

### 4. Launch the Streamlit App

```bash
streamlit run test.py
```

You’ll be able to:
- Generate target table from source
- View source-to-target mappings
- Query the transformed dataset
- Visualize data flows

---

## 📂 Project Structure

```
.
├── test.py                     # Main Streamlit app for mapping and querying
├── Preproccessing.docx         # Preprocessing logic documentation
├── Project1preprocessing.ipynb # Jupyter notebook for data exploration
├── Data Mapping_Presentation   # Final slide deck summarizing the architecture
├── source.csv                  # Raw input table (user-provided)
├── sample_data.csv             # Reference format table (user-provided)
├── target.csv                  # Generated output table (created during execution)
└── README.md                   # Project overview and instructions
```

---

## 🧠 Core Logic Breakdown

### 🔧 Preprocessing Script (`Preproccessing.docx`)
- Dynamically parses and cleans CSVs
- Detects and removes mixed data types
- Handles missing values via `SimpleImputer`
- Converts consistent date formats
- Label encodes low-cardinality categorical columns

### 🤖 Mapping Engine (`test.py`)
- Initializes LLM with a prompt template
- Uses first two rows from source and sample CSVs to understand structure
- Generates and executes transformation Python code
- Displays:
  - Mapped `target.csv`
  - Mapping dictionary
  - Natural language summary of mappings
  - NetworkX-based graph via Plotly

### 🔎 Querying Interface
- Accepts natural language queries
- Uses GPT to generate Python code that filters the data
- Displays the query result via `query_results.txt`

---

## 📈 Sample Output

**Column Mapping Example:**
```json
{
  "Name": "Patient",
  "Sex": "Gender",
  "MedicalCondition": "Diagnosis",
  "Doctor": "Physician",
  ...
}
```

**Visualization:**
- Blue nodes = Source columns
- Green nodes = Target columns
- Arrows show mapping/transformation relationships

---

## 👥 Contributors

- Nikita Singh  

---

## 📄 License

This project is for academic and learning purposes and is not licensed for commercial deployment.

---

## 📬 Contact

**Nikita Singh**  
📧 Email: [mail2nikita95@gmail.com] 
🔗 LinkedIn: [linkedin.com/in/nikitasingh3](https://linkedin.com/in/nikitasingh3)
