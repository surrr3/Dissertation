# Annual Leave Allocation Tool


The **Annual Leave Allocation Tool** is an automated annual leave scheduling solution that uses constraint programming to allocate leave to employees at PwC UK. It aims to optimise fairness and coverage while respecting individual preferences and organisational requirements.

## Technologies Used

- Python 3.10.11
- [Google OR-Tools CP-SAT Solver](https://developers.google.com/optimization/cp) - constraint programming solver
- [Pandas](https://pandas.pydata.org/) - used for data handling
- [Numpy](https://numpy.org/) - used for mathematical operations
- [Streamlit](https://streamlit.io/) - user interface

## Installation

1. Clone the repository:
   ```bash
   # On Windows:
   git clone https://github.com/surrr3/Dissertation.git

   cd Dissertation
   ```


2. (Recommended) Create and activate a virtual environment:

    ```bash
    python -m venv venv 
    
    # On Windows: 
    venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash

    pip install -r requirements.txt

    ```

## Run the Solution

1. Navigate to the correct folder:
    ```bash
    # On Windows:
    cd finalSolution
    ```

2. Start the Streamlit UI:
    ```bash
    # On Windows:
    streamlit run streamlit_interface.py
    ```

3. A browser window should automatically open, displaying the app's UI. From there, you can generate or upload data, configure parameters and run the solver. 

## How to use the Tool:

When the browser window starts up, you should see a blank page and a sidebar with controls, similar to the image below:

Using the sidebar, you can either generate data to use within the model, or input a file. 

Once data has been generated or imported, it should appear in the main part of the page, similar to the image below:

You can edit the tables displayed on the page, and edit all values except the number of employees and number of days.

Once you are happy with the parameters, navigate to the Output tab. Here, you can configure additional project parameters and also add a previous annual leave allocation that has been made by the model, if you woult like this to be taken into consideration.

Once you are happy, press the "Solve" button.

After pressing this button, the solver should generate one or more solutions. You can scroll through these solutions and pick the one you like the most. If you want to edit any solutions, click the "Edit Solution" button. Reasons are also given for each solution. 

After editing a solution, clicking "Export" will download a CSV file of the solution for future use. 

