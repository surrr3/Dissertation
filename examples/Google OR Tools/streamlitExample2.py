import pandas as pd
from math import ceil
import streamlit as st

from ALAllocation import *


####################################################
# Streamlit app

st.set_page_config(layout="wide")

if "problem" not in st.session_state:
    st.session_state.problem = None

def create_problem():

    # try:
    st.session_state.solution_exists = False
    st.session_state.problem = Problem(num_staff=st.session_state.num_staff, num_days=st.session_state.num_days, leaveAllowanceLimits=st.session_state.leave_entitlement_range, quotaLimits=st.session_state.quotas_range, preferencePercentage=st.session_state.preference_percentage, startDate=st.session_state.modelling_start_date, staffGradesChoices=st.session_state.grades_selection)
        
    # except Exception as e:
    #     st.error(f"Error creating problem: {e}")

def create_problem_from_file(file):

    try:
        lines = file.read().decode("utf-8").splitlines()
        data = [line.split(",") for line in lines]

        st.session_state.problem = Problem(int(data[0][0]), int(data[1][0]), leaveAllowanceArray=[int(x) for x in data[2]], quotaArray=[float(x) for x in data[3]], preferenceMatrix=[[int(x) for x in row] for row in data[4:]])

        st.write(st.session_state.problem)

    except Exception as e:
        st.error(f"Error reading file: {e}")
    pass

def update_matrix(old, changes, employee=True):
    
    new = old.copy()

    for i, change in changes.items():
        for j, value in change.items():
            new.at[f"Employee {i+1}" if employee else grades_reverse[i],j] = value

    return new


def get_previous_allocation(file):

    prev_allocation = FileParser(file)
    prev_allocation.read_file()
    return pd.DataFrame(prev_allocation.matrix, columns=[i.strftime('%d/%m/%Y') for i in get_dates(st.session_state.prev_start_date, prev_allocation.num_days)], index=[f"Employee {i+1}" for i in range(prev_allocation.num_staff)])



def get_streamlit_parameters():


    preferences_updated = update_matrix(st.session_state.preference_matrix_df, st.session_state.preference_matrix["edited_rows"]).astype(int).to_numpy()

    allowances_grades_updated = update_matrix(st.session_state.ag_df, st.session_state.allowances_grades["edited_rows"])

    print("\n")
    print(allowances_grades_updated)

    allowances = allowances_grades_updated["Leave Allowance (Days)"].values
    grades = allowances_grades_updated["Grade"].values

    daily_quotas_updated = update_matrix(st.session_state.dq_df, st.session_state.daily_quotas["edited_rows"], employee=False).to_numpy()
    

    new_problem = Problem(num_staff=st.session_state.problem.num_staff, num_days=st.session_state.problem.num_days, quotaArray=daily_quotas_updated, leaveAllowanceArray=allowances, preferenceMatrix=preferences_updated, startDate=st.session_state.modelling_start_date, staffGradesChoices=grades)

    if st.session_state.prev_file is not None:
        prev_allocation = FileParser(st.session_state.prev_file)
        prev_allocation.read_file()

        new_problem.add_past_allocation(prev_allocation, st.session_state.prev_start_date)
    
    print(new_problem)

    return new_problem



def generateStreamlitSolution():
    st.session_state.solution_exists = True

    st.session_state.problem = get_streamlit_parameters()

    st.session_state.problem.set_weightings(st.session_state.preference_weighting, st.session_state.consecutive_days_weighting, st.session_state.remaining_leave_allowance_weighting)

    st.session_state.problem.createModel()

    print(st.session_state.problem)
    st.session_state.solution_array = []
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


with st.sidebar:
        selection = st.segmented_control("Input Mode", ["Generate Input", "Input from File"], selection_mode="single", default="Generate Input")

        date = st.date_input("Start Date", value="today", format="DD/MM/YYYY", key="modelling_start_date")

        
        placeholder = st.empty()

        if selection == "Input from File":
            with placeholder.container():
                uploaded_file = st.file_uploader("Choose a CSV file.", type="csv")
                if uploaded_file is not None:
                    st.session_state.problem = None
                    st.session_state.solution_exists = False
                    
                    create_problem_from_file(uploaded_file)

        else:
            with placeholder.container():

                with st.form(key="input_parameters"):
                    st.write("Input Parameters")

                    col1, col2 = st.columns(2)

                    with col1:

                        num_staff = st.number_input("Number of Staff", min_value=1, value=5, key="num_staff")

                    with col2:
                        num_days = st.number_input("Number of Days", min_value=1, value=5, key="num_days")

                    st.multiselect("Staff Grades",
                    ["Associate", "Senior Associate", "Manager", "Senior Manager", "Director"],
                    default=["Associate"], key="grades_selection")

                    leave_entitlement_range = st.slider("Leave Entitlement Range", 0, 25, (5, 25), step=1, key="leave_entitlement_range")

                    quotas_range = st.slider("Daily Leave Quotas Range (%)", 0, 100, (5, 25), step=1, key="quotas_range")

                    preference_percentage = st.number_input("Percentage of Leave for Generated Preference Matrix", min_value=1, value=9, key="preference_percentage")

                    
                    submit_button = st.form_submit_button(label="Generate Data", on_click=create_problem)



tab1, tab2 = st.tabs(["Data", "Output"])

with tab1:
    
    if st.session_state.problem:
        col1, col2 = st.columns([2,3])
        with col1:
            st.metric(label="Employees", value=st.session_state.problem.num_staff)
            st.markdown("##### Staff Leave Allowances")

            # allowances_grades = st.session_state.problem.staff_leave_allowance.copy()

            allowances_grades = pd.DataFrame(st.session_state.problem.staff_leave_allowance, columns=["Leave Allowance (Days)"], index=[f"Employee {i+1}" for i in range(st.session_state.problem.num_staff)])

            allowances_grades["Grade"] = [grades_reverse[x] for x in st.session_state.problem.staff_grades]

            st.session_state.ag_df = allowances_grades.copy()

            st.data_editor(allowances_grades, key="allowances_grades")
            
        with col2:
            st.metric(label="Days", value=st.session_state.problem.num_days)
            st.markdown("##### Daily Quotas")

            st.session_state.dq_df = pd.DataFrame(st.session_state.problem.daily_quotas, columns=[d.strftime('%d/%m/%Y') for d in st.session_state.problem.dates], index=[x for x in grades.keys()])
            st.data_editor(st.session_state.dq_df, key="daily_quotas")
        
        st.markdown("#### Preference Matrix")

        st.session_state.preference_matrix_df = pd.DataFrame(st.session_state.problem.preference_matrix, columns=[d.strftime('%d/%m/%Y') for d in st.session_state.problem.dates], index=[f"Employee {i+1}" for i in range(st.session_state.problem.num_staff)])

        st.data_editor(st.session_state.preference_matrix_df, key="preference_matrix")

    else:
        st.write("No model data generated.")

    pass


with tab2:

    if not st.session_state.problem:
        st.write("No model data generated.")

    else:

        if st.session_state.solution_exists == False:

            st.markdown("#### Solution Parameters")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.number_input("Preference Weighting", min_value=0, value=2, key="preference_weighting")            
            with col2:
                st.number_input("Consecutive Days Weighting", min_value=0, value=5, key="consecutive_days_weighting")
            with col3:
                st.number_input("Remaining Leave Allowance Weighting", min_value=0, value=1, key="remaining_leave_allowance_weighting")

            with st.expander("Add A Previous Allocation"):

                prev_start_date = st.date_input("Start Date of Previous Allocation", value="today", format="DD/MM/YYYY", key="prev_start_date")

                prev_file = st.file_uploader("Upload a CSV file with previous allocation data. Must have the same dimensions as the current allocation", type="csv", key="prev_file")


                if prev_file is not None:

                    st.dataframe(get_previous_allocation(prev_file).style.applymap(highlight_past_allocation))

                    


            st.button("Solve Model", on_click=generateStreamlitSolution)

        else:

            for i, solution in enumerate(st.session_state.solution_array):
                st.markdown(f"#### Solution {i+1}")
                st.dataframe(solution[0], selection_mode=["single-row", "single-column"],on_select="rerun")

                # with st.expander("Decisions made:"):

                    # explain_solution(solution[1], st.session_state.problem)
                              
    pass



