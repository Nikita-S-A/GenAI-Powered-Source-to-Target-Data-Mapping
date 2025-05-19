import streamlit as st
import pandas as pd
import re
import sys
import io
import networkx as nx
import plotly.graph_objects as go
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(layout="wide")

OpenAI_API_KEY = 'Input Key'

# Load the CSV files
source_df = pd.read_csv('source.csv')
sample_df = pd.read_csv('sample_data.csv')

template = """You are an assistant to generate code.

Let's think step by step

1. You are given 2 tables. Source and Sample.
2. Task is to generate a target table which has exactly the same number of columns as the sample table and the same number of rows as the Source table.
3. For each column in the sample table, identify which column matches from Source table and find the transformation needed from source to sample table.
4. Use pandas in built functions or regex and transform the column into sample table format.
5. Always transform dates into mm/dd/yyyy format.
6. Do not change the Source and sample table values. Instead, find the transformations and apply it on the target table.
7. Do not perform merge or concat, as the tables are huge.
8. The column names in the sample table might not match exactly in the Source table. Identify the columns based on the column values.
9. Generate a column_mapping dictionary to show which columns in Source match with sample table columns.
10. Generate python code to create target table by reading source.csv and sample_data.csv.
11. Save the transformed DataFrame as target.csv.

Few rows of Source and Sample tables:

Source - {source_row}
Sample - {sample_row}

Python Code:
"""

# Create a prompt template
prompt = PromptTemplate(template=template, input_variables=["source_row", "sample_row"])

# Initialize the LLM with your API key
llm = ChatOpenAI(openai_api_key=OpenAI_API_KEY, model="gpt-4")
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# Convert the first 2 rows of each dataframe to JSON for the prompt
source_row = source_df.iloc[:2].to_json()
sample_row = sample_df.iloc[:2].to_json()

option = st.sidebar.radio("Menu", ["Mapping", "Querying"])
if option == "Mapping":

    st.title("Source to Target Data Mapping")

    # User input for generating the target table
    if st.button("Generate Target Table"):
        # Generate the response
        response = llm_chain.run({"source_row": source_row, "sample_row": sample_row})

        # Extract the Python code block from the response
        code_block = re.search(r"```python\n(.*?)```", response, re.DOTALL)

        if code_block:
            code = code_block.group(1)
            # Save the extracted code to a Python file
            with open("generated_code.py", "w") as file:
                file.write(code)

            # Execute the generated code
            exec(code)

            # Load the generated target dataframe
            target_df = pd.read_csv('target.csv')

            # Display the source, sample, and target dataframes
            st.write("### Source ")
            st.dataframe(source_df.head(10))  # Display only the first 10 rows

            st.write("### Target ")
            st.dataframe(target_df.head(10))  # Display only the first 10 rows

            # Extract the column mapping from the generated code
            column_mapping = re.search(r"column_mapping\s*=\s*{(.*?)}", code, re.DOTALL)
            

            if column_mapping:
                column_mapping_str = "{" + column_mapping.group(1) + "}"
                column_mapping = eval(column_mapping_str)

                # Display the column mapping as a table
                st.write("### Column Mapping")
                column_mapping_df = pd.DataFrame(list(column_mapping.items()), columns=["Source Column", "Target Column"])
                st.dataframe(column_mapping_df)

                # Make an API call to GPT for generating a summary
                client = OpenAI(api_key=OpenAI_API_KEY)
                user_content = f"Provide a detailed summary of the following column mapping in layman terms: {column_mapping_str}"
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant to explain data mappings.return explanation only."},
                        {"role": "user", "content": user_content},
                    ]
                )

               

                # Display the summary
                
                st.text(response.choices[0].message.content)

                # Add a button to generate summary
                if st.button("Generate Summary"):
                    

                    # Visualize the mapping using Plotly
                    G = nx.DiGraph()

                    # Add nodes and edges based on the column mapping
                    for source_col, target_col in column_mapping.items():
                        G.add_node(source_col)
                        G.add_node(target_col)
                        G.add_edge(source_col, target_col)

                    pos = nx.spring_layout(G)

                    # Extracting edges and nodes for plotly
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.append(x0)
                        edge_x.append(x1)
                        edge_x.append(None)
                        edge_y.append(y0)
                        edge_y.append(y1)
                        edge_y.append(None)

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=2, color='#888'),
                        hoverinfo='none',
                        mode='lines')

                    node_x = []
                    node_y = []
                    node_color = []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_color.append('lightblue' if node in column_mapping else 'lightgreen')

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=[node for node in G.nodes()],
                        textposition="top center",
                        hoverinfo='text',
                        marker=dict(
                            showscale=False,
                            color=node_color,
                            size=20,
                            line_width=2))

                    # Create custom legend entries
                    legend_trace_source = go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=20, color='lightblue'),
                        legendgroup='Source',
                        showlegend=True,
                        name='Source')

                    legend_trace_target = go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=20, color='lightgreen'),
                        legendgroup='Target',
                        showlegend=True,
                        name='Target')

                    fig = go.Figure(data=[edge_trace, node_trace, legend_trace_source, legend_trace_target],
                                    layout=go.Layout(
                                        showlegend=True,
                                        hovermode='closest',
                                        margin=dict(b=0, l=0, r=0, t=40),
                                        xaxis=dict(showgrid=False, zeroline=False),
                                        yaxis=dict(showgrid=False, zeroline=False))
                                    )
                    st.plotly_chart(fig)
            else:
                st.error("Column mapping not found in the generated code")
        else:
            st.error("No code block found in the response")

elif option == "Querying":
    st.title("Query Target Table")

    user_query = st.text_input("Enter your query:")
    if st.button("Submit Query"):

        client = OpenAI(api_key=OpenAI_API_KEY)
        user_content = f"Generate python code using pandas to answer the query {user_query}. And store the results in a file called query_results.txt"
        response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant to generate code. data is in target.csv. columns are Patient,Age,Gender,Blood Type,Diagnosis,Date of Admission,Physician,Hospital,Health Plan Provider,Billing Amount,Room Number,Admission Type,Date of Discharge,Medication,Lab Results"},
                    {"role": "user", "content": user_content},
                ]
            )
        # st.write(response)
        code_block = re.search(r"```python\n(.*?)```", response.choices[0].message.content, re.DOTALL)

        if code_block:
            code = code_block.group(1)
            exec(code)

        try:
            with open('query_results.txt','r') as file:
                summary = file.read()
                st.write("Result")
                st.text(summary)

        except:
            st.error('Error in generated code.')