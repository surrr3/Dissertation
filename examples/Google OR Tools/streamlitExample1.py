import random
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from math import ceil
import time

class Problem:

    daily_quotas = None
    staff_leave_allowance = None
    preference_matrix = None
    model = None
    solver = None
    status = None


    def __init__(self, num_staff: int, num_days: int, quotaLimits: tuple = None, quotaArray: list = None, leaveAllowanceLimits: tuple = None, leaveAllowanceArray: list = None, preferencePercentage:int = None, preferenceMatrix: list = None):
        self.num_staff = num_staff
        self.num_days = num_days            

        if quotaLimits:
            self.daily_quotas = [((random.randrange(quotaLimits[0], quotaLimits[1]) / 100)) for i in range(self.num_days)] 

        elif quotaArray:
            if not len(quotaArray) == self.num_days:
                raise ValueError("Quota array must have the same length as the number of days in the problem")
            
            self.daily_quotas = quotaArray

        if leaveAllowanceLimits:
            self.staff_leave_allowance = [random.randint(leaveAllowanceLimits[0], leaveAllowanceLimits[1]) for i in range(self.num_staff)]

        elif leaveAllowanceArray:
            if not len(leaveAllowanceArray) == self.num_staff:
                raise ValueError("Leave allowance array must have the same length as the number of staff in the problem")
            
            self.staff_leave_allowance = leaveAllowanceArray

        if not preferenceMatrix:
            self.generate_preference_matrix(preferencePercentage)
        else:    
            self.preference_matrix = np.array(preferenceMatrix).reshape(self.num_staff, self.num_days)

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

    def to_csv(self, filename):
        ## ADD OTHER METRICS TO FILE
        df = pd.DataFrame(self.preference_matrix)
        df.to_csv(filename, index=False, header=False)

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

        objective_1 = sum(self.preference_matrix[e][d] * self.l[e, d] for e in range(self.num_staff) for d in range(self.num_days))

        objective_1_weighting = 5

        consecutive_days_off = {}

        for e in range(self.num_staff):
            for d in range(self.num_days-1):
                consecutive_days_off[(e, d)] = self.model.new_bool_var(f"consecutive_days_off_{e}_{d}")

        for e in range(self.num_staff):
            for d in range(self.num_days-1):
                self.model.add_multiplication_equality(consecutive_days_off[(e, d)], [self.l[e, d], self.l[e, d+1]])

        objective_2 = sum(consecutive_days_off[(e, d)] for e in range(self.num_staff) for d in range(self.num_days-1))

        objective_2_weighting = 4


        objective_3 = sum(self.staff_leave_allowance[e] * self.l[e, d] for e in range(self.num_staff) for d in range(self.num_days))
        objective_3_weighting = 2


        self.model.maximize(
                (objective_1_weighting * objective_1) + 
                (objective_2_weighting * objective_2) +
                (objective_3_weighting * objective_3)
            )   

        pass

    def solve(self):
        self.solver = cp_model.CpSolver()
        self.solver.parameters.enumerate_all_solutions = True
        self.status = self.solver.Solve(self.model)

    def getResults(self):
        if self.status == cp_model.OPTIMAL or self.status == cp_model.FEASIBLE:

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
        st.session_state.results = self.getResults()

def highlight_diff(data, other, color='#ff616b'):
    attr = f'background-color: {color}'
    return pd.DataFrame(np.where(data.ne(other), attr, ''),
                        index=data.index, columns=data.columns)

def highlight_cells(val):
    color = '#1eff00' if val == 1 else ''
    return 'color: %s' % color


import streamlit as st
st.set_page_config(layout="wide")

if "problem" not in st.session_state:
    st.session_state.problem = None

def create_problem():
    st.session_state.problem = Problem(num_staff=st.session_state.num_staff, num_days=st.session_state.num_days, leaveAllowanceLimits=st.session_state.leave_entitlement_range, quotaLimits=st.session_state.quotas_range, preferencePercentage=st.session_state.preference_percentage)
    st.session_state.submitted = True


with st.sidebar:
        selection = st.segmented_control("Input Mode", ["Random Input", "Input from File", "Sample Datasets"], selection_mode="single")
        
        placeholder = st.empty()

        if selection == "Input from File":
            with placeholder.container():
                uploaded_file = st.file_uploader("Choose a file")
        elif selection == "Sample Datasets":
            with placeholder.container():
                sample_data = st.selectbox("Select Sample Dataset", ["Sample 1", "Sample 2", "Sample 3"])

                with st.empty().container():
                    if sample_data == "Sample 1":
                        st.write("Sample 1")
                        st.write("Sample 1")
                    elif sample_data == "Sample 2":
                        st.write("Sample 2")
                    elif sample_data == "Sample 3":
                        st.write("Sample 3")
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

    else:
        st.write("No model data generated.")

    pass

with tab2:
    if not st.session_state.problem:
        st.write("No model data generated.")

    else:

        if "results" not in st.session_state:
            st.markdown("# Solve the model")
            st.button("Solve", on_click=st.session_state.problem.generateSolution)

        else:
            st.dataframe(st.session_state.results)
    pass