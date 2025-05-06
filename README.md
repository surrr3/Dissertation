# Annual Leave Allocation Tool


The **Annual Leave Allocation Tool** is an automated annual leave scheduling solution that uses constraint programming to allocate leave to employees at PwC UK. It aims to optimise fairness and coverage while respecting individual preferences and organisational requirements.

## Technologies Used

- Python 3.10.11
- [Google OR-Tools CP-SAT Solver](https://developers.google.com/optimization/cp) - The chosen Constraint Programming solver.
- [Pandas](https://pandas.pydata.org/) - Used for displaying solutions as DataFrames.
- [Numpy](https://numpy.org/) - Used for generating datasets.
- [Streamlit](https://streamlit.io/) - The library used to create the user interface.

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

3. A browser window should automatically open, displaying the app. From there, you can generate or upload data, configure parameters and run the solver. The URL of the app should be [http://localhost:8501](http://localhost:8501)


## How to use the Tool:

When the browser window starts up, you should see a blank page and a sidebar with controls, similar to the image below:

![Screenshot 2025-04-30 182032](https://github.com/user-attachments/assets/92a97403-a113-40c4-a044-93cca9f43bc3)


Using the sidebar, you can either generate data to use within the model, or input a file.

Once data has been generated or imported, it should appear in the main part of the page, similar to the image below:

![Screenshot 2025-04-30 182106](https://github.com/user-attachments/assets/f942ea5c-de70-4eb9-b268-35bea4e42b95)

You can edit the tables displayed on the page, and edit all values except the number of employees and number of days.

Once you are happy with the parameters, navigate to the Output tab. Here, you can configure additional project parameters and also add a previous annual leave allocation that has been made by the model, if you would like this to be taken into consideration.

Once you are happy, press the "Solve" button.

![Screenshot 2025-04-30 182217](https://github.com/user-attachments/assets/83de08b5-c2ba-4ceb-9e9c-b8dafcdde9ca)


After pressing this button, the solver should generate one or more solutions. You can scroll through these solutions and pick the one you like the most. If you want to edit any solutions, click the "Edit Matrix X" button. Reasons are also given for each solution. 

![Screenshot 2025-04-30 182302](https://github.com/user-attachments/assets/5da5f88e-6a59-4391-b99a-a9201478faa6)


After editing a solution, clicking "Download CSV" will download a CSV file of the solution for future use. 

![Screenshot 2025-04-30 182319](https://github.com/user-attachments/assets/5542effd-9563-4de6-9b27-9e46140ae993)
