import random
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from math import ceil
import streamlit as st
import time


# Function to highlight differences in DataFrame
def highlight_diff(data, other, color='#ff616b'):
    attr = f'background-color: {color}'
    return pd.DataFrame(np.where(data.ne(other), attr, ''),
                        index=data.index, columns=data.columns)

# Function to highlight cells based on value
def highlight_cells(val):
    color = '#1eff00' if val == 1 else ''
    return 'color: %s' % color



class SolutionPrinter(cp_model.CpSolverSolutionCallback):



    def __init__(self, num_staff, num_days, l, preferences, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._num_staff = num_staff
        self._num_days = num_days
        self._leave = l
        self._solution_count = 0
        self._solution_limit = limit
        self._preference_matrix = preferences



    def on_solution_callback(self):

        self._solution_count += 1    

        solutionArray = [[self.value(self._leave[(s, d)]) for d in range(self._num_days)] for s in range(self._num_staff)]

        df = pd.DataFrame(solutionArray, columns=[f'Day {d+1}' for d in range(self._num_days)], index=[f'Employee {x+1}' for x in range(self._num_staff)])

        df2 = pd.DataFrame(self._preference_matrix, columns=[f'Day {d+1}' for d in range(self._num_days)], index=[f'Employee {x+1}' for x in range(self._num_staff)])
        
        df_styled = df.style.applymap(highlight_cells).apply(highlight_diff, axis=None, other=df2)

        print(f"Hamming Distance: {self.getHammingDistance(solutionArray, self._preference_matrix)}")



        st.session_state.solution_array.append((df_styled,df))
        print(st.session_state.solution_array)


        if self._solution_count >= self._solution_limit:
            print(f"Stop search after {self._solution_limit} solutions")
            self.stop_search()

    def getMetrics(self, allocation, preferences):

        hamming_distance = np.sum(np.abs(np.array(allocation) - np.array(preferences)))
        return hamming_distance



    def solutionCount(self):
        return self._solution_count
    
class Problem:

    daily_quotas = None
    staff_leave_allowance = None
    preference_matrix = None
    model = None
    solver = None
    solution = None


    # Constructor for the Problem class
    # num_staff: number of staff members
    # num_days: number of days in the problem
    # For quota, can choose between defining an array or giving limits for data to be randomly generated
    # quotaLimits: tuple of min and max quota limits (optional)
    # quotaArray: list of quotas for each day (optional)
    # For leave allowance, can choose between defining an array or giving limits for data to be randomly generated
    # leaveAllowanceLimits: tuple of min and max leave allowance limits (optional)
    # leaveAllowanceArray: list of leave allowances for each staff member (optional)
    # for preference, can choose between defining a matrix or giving percentage of requests for data to be randomly generated
    # preferencePercentage: percentage of preference for each staff member (optional)
    # preferenceMatrix: list of preferences for each staff member (optional)
    def __init__(self, num_staff: int, num_days: int, quotaLimits: tuple = None, quotaArray: list = None, leaveAllowanceLimits: tuple = None, leaveAllowanceArray: list = None, preferencePercentage:int = None, preferenceMatrix: list = None):

        # get number of staff and number of days
        self.num_staff = num_staff
        self.num_days = num_days            

        # get daily quotas, either randomly generate or get from array
        if quotaLimits:
            self.daily_quotas = [((random.randrange(quotaLimits[0], quotaLimits[1]) / 100)) for i in range(self.num_days)] 

        elif quotaArray:
            if not len(quotaArray) == self.num_days:
                raise ValueError("Quota array must have the same length as the number of days in the problem")
            
            self.daily_quotas = quotaArray

        # get staff leave allowance, either randomly generate or get from array
        if leaveAllowanceLimits:
            self.staff_leave_allowance = [random.randint(leaveAllowanceLimits[0], leaveAllowanceLimits[1]) for i in range(self.num_staff)]

        elif leaveAllowanceArray:
            if not len(leaveAllowanceArray) == self.num_staff:
                raise ValueError("Leave allowance array must have the same length as the number of staff in the problem")
            
            self.staff_leave_allowance = leaveAllowanceArray

        # get preference matrix, either randomly generate or get from array
        if not preferenceMatrix:
            self.generate_preference_matrix(preferencePercentage)
        else:    
            self.preference_matrix = np.array(preferenceMatrix).reshape(self.num_staff, self.num_days)

        # check that all values are defined
        if not self.daily_quotas:
            raise ValueError("Quotas must be defined")
        
        if not self.staff_leave_allowance:
            raise ValueError("Staff leave allowance must be defined")
        
        if self.preference_matrix.any() == None:
            raise ValueError("Preference matrix must be defined")
        
    def __str__(self):
        return f"{self.num_staff} staff, {self.num_days} days,\nquotas:\n{self.daily_quotas},\nstaff leave allowance:\n{self.staff_leave_allowance}\npreference matrix:\n{self.preference_matrix}"
        

    def generate_preference_matrix(self, preferencePercentage):
        num_ones = int(self.num_staff * self.num_days * (preferencePercentage / 100))
        num_zeros = (self.num_staff * self.num_days) - num_ones
        array = np.array([1] * num_ones + [0] * num_zeros)
        np.random.shuffle(array)
        self.preference_matrix = array.reshape(self.num_staff,self.num_days)

    def dataset_to_csv(self):
        data = [[self.num_days], [self.num_staff], self.staff_leave_allowance, self.daily_quotas]

        data.extend(self.preference_matrix.tolist())

        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False, header=False)
        return csv_data

    # def from_csv(self, filename):
        # df = pd.read_csv(filename, header=None)

    def createModel(self):
        self.model = cp_model.CpModel()

        # Matrix for leave granted
        # if employee e has day d off, then L[e,d] = 1
        self.l = {}
        for e in range(self.num_staff):
            for d in range(self.num_days):
                self.l[(e, d)] = self.model.new_bool_var(f"L_{e}_d{d}")

        # Staff leave allowance not exceeded
        for e in range(self.num_staff):
            self.model.Add(sum(self.l[e, d] for d in range(self.num_days)) <= self.staff_leave_allowance[e])
        
        # daily leave quota not exceeded
        # calculate number of staff * staff_limit for each day
        for d in range(self.num_days):
            self.model.Add(sum(self.l[e, d] for e in range(self.num_staff)) <= ceil(self.num_staff * self.daily_quotas[d]))

        #############################################
        # Objective function:
        # maximise preferences satisfied
        objective_1 = sum(self.preference_matrix[e][d] * self.l[e, d] for e in range(self.num_staff) for d in range(self.num_days))

        objective_1_weighting = 5

        # maximise number of consecutive days off
        consecutive_days_off = {}

        for e in range(self.num_staff):
            for d in range(self.num_days-1):
                consecutive_days_off[(e, d)] = self.model.new_bool_var(f"consecutive_days_off_{e}_{d}")

        for e in range(self.num_staff):
            for d in range(self.num_days-1):
                self.model.add_multiplication_equality(consecutive_days_off[(e, d)], [self.l[e, d], self.l[e, d+1]])

        objective_2 = sum(consecutive_days_off[(e, d)] for e in range(self.num_staff) for d in range(self.num_days-1))

        objective_2_weighting = 4


        # give preference to days off with higher leave allowance
        objective_3 = sum(self.staff_leave_allowance[e] * self.l[e, d] for e in range(self.num_staff) for d in range(self.num_days))
        objective_3_weighting = 2

        # Objective function
        self.model.maximize(
                (objective_1_weighting * objective_1) + 
                (objective_2_weighting * objective_2) +
                (objective_3_weighting * objective_3)
            )   

        pass

    # Solve the model
    def solve(self):
        self.solver = cp_model.CpSolver()
        self.solver.parameters.enumerate_all_solutions = True
        self.solution = self.solver.Solve(self.model)

    # get the results of the model
    def getResults(self):
        if self.solution == cp_model.OPTIMAL or self.solution == cp_model.FEASIBLE:

            solutionArray = [[self.solver.Value(self.l[s,d]) for d in range(self.num_days)] for s in range(self.num_staff)]

            df = pd.DataFrame(solutionArray, columns=[f'Day {d+1}' for d in range(self.num_days)], index=[f'Employee {x+1}' for x in range(self.num_staff)])

            df2 = pd.DataFrame(self.preference_matrix, columns=[f'Day {d+1}' for d in range(self.num_days)], index=[f'Employee {x+1}' for x in range(self.num_staff)])


            df_styled = df.style.applymap(highlight_cells).apply(highlight_diff, axis=None, other=df2)

            return df_styled

        else:
            print("No feasible solution found.")

    def generateSolution(self):
        self.createModel()
        self.solve()

    def list_all_solutions(self):

        solution_limit = 10
        solution_printer = SolutionPrinter(self.num_staff, self.num_days, self.l, self.preference_matrix, solution_limit)
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True

        solver.Solve(self.model, solution_printer)       



# problem = Problem(5, 7, quotaLimits=(0, 100), leaveAllowanceLimits=(0, 3), preferencePercentage=50)
# print(problem)
# problem.generateSolution()
# problem.getResults()
# problem.list_all_solutions()

####################################################
# Streamlit app

st.set_page_config(layout="wide")

if "problem" not in st.session_state:
    st.session_state.problem = None

def create_problem():
    
    st.session_state.solution_exists = False
    st.session_state.problem = Problem(num_staff=st.session_state.num_staff, num_days=st.session_state.num_days, leaveAllowanceLimits=st.session_state.leave_entitlement_range, quotaLimits=st.session_state.quotas_range, preferencePercentage=st.session_state.preference_percentage)
    st.session_state.submitted = True

def create_problem_from_sample(num_staff, num_days, leaveAllowanceArray, quotaArray, preferenceMatrix):

    st.session_state.solution_exists = False
    st.session_state.problem = None
    st.session_state.problem = Problem(num_staff=num_staff, num_days=num_days, leaveAllowanceArray=leaveAllowanceArray, quotaArray=quotaArray, preferenceMatrix=preferenceMatrix)
    
    st.session_state.submitted = True



def get_data_from_file(file):
    try:
        lines = file.read().decode("utf-8").splitlines()
        data = [line.split(",") for line in lines]

        st.session_state.problem = Problem(int(data[0][0]), int(data[1][0]), leaveAllowanceArray=[int(x) for x in data[2]], quotaArray=[float(x) for x in data[3]], preferenceMatrix=[[int(x) for x in row] for row in data[4:]])

        st.write(st.session_state.problem)

    except Exception as e:
        st.error(f"Error reading file: {e}")
    pass

with st.sidebar:
        selection = st.segmented_control("Input Mode", ["Random Input", "Input from File", "Sample Datasets"], selection_mode="single", default="Random Input")
        
        placeholder = st.empty()

        if selection == "Input from File":
            with placeholder.container():
                uploaded_file = st.file_uploader("Choose a CSV file.", type="csv")
                if uploaded_file is not None:
                    st.session_state.problem = None
                    st.session_state.solution_exists = False
                    
                    get_data_from_file(uploaded_file)

        elif selection == "Sample Datasets":
            with placeholder.container():
                sample_data = st.selectbox("Select Sample Dataset", ["Sample 1", "Sample 2", "Sample 3"])

                with st.empty().container():
                    if sample_data == "Sample 1":
                        st.markdown('''
                        ### Sample 1:
                                    
                        - **Staff:** 5

                        - **Days:** 5
                                    
                        - **Leave Allowance:** [2, 2, 1, 5, 0]
                                    
                        - **Daily Quotas:** [0.5, 0.4, 0.2, 0.2, 0.3]
                                    
                        - **Preference Matrix:** ''')

                        s1_preference_matrix = [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]

                        st.dataframe(pd.DataFrame(s1_preference_matrix, columns=[f'Day {d+1}' for d in range(5)], index=[f'Employee {x+1}' for x in range(5)]))

                        st.button("Generate Data",on_click=create_problem_from_sample, args=(5, 5, [2, 2, 1, 5, 0], [0.5, 0.4, 0.2, 0.2, 0.3], s1_preference_matrix ))

                    elif sample_data == "Sample 2":
                        st.markdown('''
                        ### Sample 2:
                                    
                        - **Staff:** 5

                        - **Days:** 10
                                    
                        - **Leave Allowance:** [2, 2, 1, 5, 0]
                                    
                        - **Daily Quotas:** [0.5, 0.4, 0.2, 0.2, 0.3, 0.5, 0.4, 0.2, 0.3, 0.1]
                                    
                        - **Preference Matrix:** ''')
                        st.dataframe(pd.DataFrame([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], columns=[f'Day {d+1}' for d in range(5)], index=[f'Employee {x+1}' for x in range(5)]))
                    elif sample_data == "Sample 3":
                        st.markdown('''
                        ### Sample 1:
                                    
                        - **Staff:** 5

                        - **Days:** 10
                                    
                        - **Leave Allowance:** [2, 2, 1, 5, 0]
                                    
                        - **Daily Quotas:** [0.5, 0.4, 0.2, 0.2, 0.3, 0.5, 0.4, 0.2, 0.3, 0.1]
                                    
                        - **Preference Matrix:** ''')
                        st.dataframe(pd.DataFrame([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], columns=[f'Day {d+1}' for d in range(5)], index=[f'Employee {x+1}' for x in range(5)]))
        else:
            with placeholder.container():

                with st.form(key="input_parameters"):
                    st.write("Input Parameters")

                    num_staff = st.number_input("Number of Staff", min_value=1, value=5, key="num_staff")
                    num_days = st.number_input("Number of Days", min_value=1, value=10, key="num_days")

                    leave_entitlement_range = st.slider("Leave Entitlement Range", 0, 365, (5, 10), step=1, key="leave_entitlement_range")

                    quotas_range = st.slider("Daily Leave Quotas Range", 0, 100, (5, 25), step=1, key="quotas_range")

                    preference_percentage = st.number_input("Percentage of Leave for Generated Preference Matrix", min_value=1, value=40, key="preference_percentage")

                    submit_button = st.form_submit_button(label="Generate Data", on_click=create_problem)



tab1, tab2 = st.tabs(["Data", "Output"])

with tab1:
    
    if st.session_state.problem:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Employees", value=st.session_state.problem.num_staff)
            st.markdown("##### Staff Leave Allowances")
            st.dataframe(pd.DataFrame({"Leave Allowance": st.session_state.problem.staff_leave_allowance}))
            
        with col2:
            st.metric(label="Days", value=st.session_state.problem.num_days)
            st.markdown("##### Daily Quotas")
            st.dataframe(pd.DataFrame({"Quota": st.session_state.problem.daily_quotas}))  
        
        st.markdown("#### Preference Matrix")

        st.dataframe(pd.DataFrame(st.session_state.problem.preference_matrix, columns=[f'Day {d+1}' for d in range(st.session_state.problem.num_days)], index=[f'Employee {x+1}' for x in range(st.session_state.problem.num_staff)]).style.map(highlight_cells))

        st.download_button(label="Download Data as CSV", data=st.session_state.problem.dataset_to_csv(), file_name="dataset.csv", mime="text/csv", icon=":material/download:")
    else:
        st.write("No model data generated.")

    pass

def generateStreamlitSolution():
    st.session_state.solution_array = []
    st.session_state.solution_exists = True
    st.session_state.problem.generateSolution()
    st.session_state.problem.list_all_solutions()

def explain_solution(solution: pd.DataFrame, problem: Problem ):

    for i in range(problem.num_staff):
        this_leave_allowance = problem.staff_leave_allowance[i]
        leave_taken = sum(solution.iloc[i,:])

        for j in range(problem.num_days):
            this_day_quota = problem.daily_quotas[j]
            this_day_quota_limit = ceil(problem.num_staff * this_day_quota)
            quota_taken = sum(solution.iloc[:,j])

            if solution.iloc[i,j] == 0 and problem.preference_matrix[i,j] == 1:


                if this_day_quota_limit >= quota_taken:
                    st.markdown(f"Employee {i+1} has not been granted leave on Day {j+1} because the staff leave limit for this day has been met ({quota_taken}/{this_day_quota_limit})")

                if this_leave_allowance <= leave_taken:

                    st.markdown(f"Employee {i+1} has not been granted leave on Day {j+1} because they have met their leave allowance ({leave_taken}/{this_leave_allowance})")

            
            elif solution.iloc[i,j] == 1 and problem.preference_matrix[i,j] == 0:
                st.markdown(f":orange[Employee {i+1} has been granted leave on Day {j+1} but did not request it]")

    pass

with tab2:
    if not st.session_state.problem:
        st.write("No model data generated.")

    else:

        if st.session_state.solution_exists == False:
            st.button("Solve", on_click=generateStreamlitSolution)

        else:

            for i, solution in enumerate(st.session_state.solution_array):
                st.markdown(f"#### Solution {i+1}")
                st.dataframe(solution[0], selection_mode=["single-row", "single-column"],on_select="rerun")

                with st.expander("Decisions made:"):

                    explain_solution(solution[1], st.session_state.problem)
                              
    pass



