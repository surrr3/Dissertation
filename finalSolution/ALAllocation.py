# File to store classes for the AL Allocation problem

import random
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from math import ceil
import streamlit as st
import datetime


# staff grades
grades = {
    'Associate': 0,
    'Senior Associate': 1,
    'Manager': 2,
    'Senior Manager': 3,
    'Director': 4
}
grades_reverse = {
    0: 'Associate',
    1: 'Senior Associate',
    2: 'Manager',
    3: 'Senior Manager',
    4: 'Director'
}


# Used to read and parse previous allocation or new model data from a file
class FileParser():

    num_days = None
    num_staff = None
    leave_allowance = None
    staff_grades = None
    quotas = None
    matrix = None

    def __init__(self, file):
        self.file = file

    # read the file and parse the data
    def read_file(self):

        lines = self.file.read().decode("utf-8").splitlines()
        data = [line.split(",") for line in lines]

        self.num_days = int(float(data[0][0].strip()))
        self.num_staff = int(float(data[2][0].strip()))
        self.staff_grades = [int(float(x.strip())) for x in  list(filter(None, data[4][0:]))]

        if len(self.staff_grades) != self.num_staff:
            raise ValueError("Staff grades array must have the same length as the number of staff in the problem")
        
        self.leave_allowance = [int(float(x.strip())) for x in list(filter(None, data[6][0:]))]

        if len(self.leave_allowance) != self.num_staff:
            raise ValueError("Leave allowance array must have the same length as the number of staff in the problem")
        
        self.quotas = [[float(x) for x in list(filter(None, data[8+y][0:]))] for y in range(5)]

        if len(self.quotas) != 5 or len(self.quotas[0]) != self.num_days:
            raise ValueError("Quota array must have a length of 5, and each row must have the same length as the number of days in the problem")
        
        self.matrix = [[int(float(x)) for x in row if x.strip()!=''] for row in list(filter(None,data[14:]))]

        if len(self.matrix) != self.num_staff or len(self.matrix[0]) != self.num_days:
            raise ValueError("Allocation matrix must have the same length as the number of staff and days in the problem")
        
    # create Problem object from the parsed data
    def to_Problem(self, startDate=None):

        # check that all values are defined
        if self.quotas is None:
            raise ValueError("Quotas must be defined")
                
        if self.leave_allowance is None:
            raise ValueError("Staff leave allowance must be defined")
        
        if self.matrix is None:
            raise ValueError("Preference matrix must be defined")
        
        if self.staff_grades is None:
            raise ValueError("Staff grades must be defined")

        # make problem object
        problem = Problem(self.num_staff, self.num_days, quotaArray=self.quotas, leaveAllowanceArray=self.leave_allowance, preferenceMatrix=self.matrix, startDate=startDate, staffGradesArray=self.staff_grades)

        return problem

# Function to highlight differences in DataFrame
def highlight_diff(data, other, colour='#ff616b'):
    attr = f'background-color: {colour}'
    return pd.DataFrame(np.where(data.ne(other), attr, ''),
                        index=data.index, columns=data.columns)

# Function to highlight cells based on value
def highlight_cells(val, colour='#1eff00'):
    return f'color: {colour}' if val == 1 else ''

# highlight cells that are the same as the previous allocation
def highlight_same(data, other, colour='#fcc203'):
    attr = f'background-color: {colour}'
    return pd.DataFrame(np.where(other==1, attr, ''),
                        index=data.index, columns=data.columns)

# get a list of dates, excluding weekends
def get_dates(startDate=None, num_days=None):

        # check if start date is defined, if not, set to today
        if not startDate:
            startDate = datetime.date.today()

        dates = []
        counter = 0

        # get list of dates for the number of days in the problem
        while len(dates) < num_days:
            date = startDate + datetime.timedelta(days=counter)

            # only add weekdays (mon-fri)
            if date.weekday() < 5:
                dates.append(date)
            counter += 1

        return dates


# Class to print the solution to the streamlit app
class SolutionPrinter(cp_model.CpSolverSolutionCallback):

    def __init__(self, num_staff, num_days, l, preferences, limit, startDate, leave_allowances, daily_quotas, grades, prev = None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.num_staff = num_staff
        self.num_days = num_days
        self.leave = l
        self.solution_count = 0
        self.solution_limit = limit
        self.preference_matrix = preferences
        self.prev = prev
        self.startDate = startDate
        self.dates = get_dates(self.startDate, self.num_days)
        self.leave_allowances = leave_allowances
        self.daily_quotas = daily_quotas
        self.grades = grades


    # function to run for each solution found, generates dataframe and stores it in session state
    def on_solution_callback(self):

        self.solution_count += 1    

        # print(f"Solution {self.solution_count}")

        # get the solution
        solutionArray = [[self.value(self.leave[(s, d)]) for d in range(self.num_days)] for s in range(self.num_staff)]

        # solution dataframe
        df = pd.DataFrame(solutionArray, columns=[date.strftime('%d/%m/%Y') for date in self.dates], index=[f'Employee {x+1}' for x in range(self.num_staff)])

        # preference matrix dataframe
        df2 = pd.DataFrame(self.preference_matrix, columns=[date.strftime('%d/%m/%Y') for date in self.dates], index=[f'Employee {x+1}' for x in range(self.num_staff)])

        df_styled = df.style.applymap(highlight_cells).apply(highlight_diff, axis=None, other=df2)

        if self.prev is not None:

            # previous allocation dataframe
            df3 = pd.DataFrame(self.prev, columns=[date.strftime('%d/%m/%Y') for date in self.dates], index=[f'Employee {x+1}' for x in range(self.num_staff)])

            df_styled = df_styled.apply(highlight_same, axis=None, other=df3)

        metrics = self.getMetrics(solutionArray=df)

        explanation = self.explain_solution(solution=df)

        st.session_state.solution_array.append((df_styled,df,metrics, explanation))

        # display(df_styled)
        # print(metrics)

        if self.solution_count >= self.solution_limit:
            self.stop_search()

    # get metrics for each solution
    def getMetrics(self, solutionArray):

        metrics = {}

        metrics["objective_value"] = self.ObjectiveValue()
        
        preference_leave_days_count = self.preference_matrix.sum().sum()
        assigned_leave_days_count = solutionArray.sum().sum()
        previous_leave_days_count = self.prev.sum().sum() if self.prev is not None else 0

        metrics["percentage_of_leave_granted"] = f"{((assigned_leave_days_count - previous_leave_days_count) / preference_leave_days_count) * 100:.2f}%"

        return metrics
    
    # explain the solution, return a list of reasons for each leave day not granted
    def explain_solution(self, solution):

        reasons = []


        for i in range(self.num_staff):
            this_leave_allowance = self.leave_allowances[i]
            leave_taken = sum(solution.iloc[i,:])

            for j in range(self.num_days):
                this_day_quota = self.daily_quotas[self.grades[i], j]
                
                quota_taken = sum([solution.iloc[k,j] for k in range(self.num_staff) if self.grades[k] == self.grades[i]])

                if solution.iloc[i,j] == 0 and self.preference_matrix[i,j] == 1:

                    if this_day_quota >= quota_taken:

                        reasons.append(f"**Employee {i+1}, Day {j+1}** : leave limit for grade {grades_reverse[self.grades[i]].upper()} met ({quota_taken}/{this_day_quota})")

                    if this_leave_allowance <= leave_taken:

                        reasons.append(f"**Employee {i+1}, Day {j+1}**: Leave allowance met: ({leave_taken}/{this_leave_allowance})")

        return reasons

    def solutionCount(self):
        return self._solution_count

# class to store the model and its parameters 
class Problem:

    daily_quotas = None
    daily_quotas_real = None
    staff_grades = None
    staff_leave_allowance = None
    preference_matrix = None
    model = None
    solver = None
    solution = None
    dates = None
    past_allocation = None
    objective_1_weighting = 2
    objective_2_weighting = 5
    objective_3_weighting = 1
    time_limit = 10
    runtime = 0


    '''
        num_staff: number of staff members
        num_days: number of days in the problem
        For quota, can choose between defining an array or giving limits for data to be randomly generated
        quotaLimits: tuple of min and max quota limits (optional)
        quotaArray: list of quotas for each day (optional)
        For leave allowance, can choose between defining an array or giving limits for data to be randomly generated
        leaveAllowanceLimits: tuple of min and max leave allowance limits (optional)
        leaveAllowanceArray: list of leave allowances for each staff member (optional)
        for preference, can choose between defining a matrix or giving percentage of requests for data to be randomly generated
        preferencePercentage: percentage of preference for each staff member (optional)
        preferenceMatrix: list of preferences for each staff member (optional)
        startDate: start date to model from (optional)
        staffGradesArray: list of staff grades for each staff member (optional)
        staffGradesChoices: list of staff grades to randomly generate from (optional)
    '''
    
    def __init__(self, num_staff: int, num_days: int, quotaLimits: tuple = None, quotaArray: list = None, leaveAllowanceLimits: tuple = None, leaveAllowanceArray: list = None, preferencePercentage:int = None, preferenceMatrix: list = None, startDate: datetime.date = None, staffGradesArray: list = None, staffGradesChoices: list = None ):

        self.num_staff = num_staff
        self.num_days = num_days
        self.startDate = startDate

        self.dates = get_dates(self.startDate, self.num_days)

        # get staff grades, either randomly generate or get from array
        if staffGradesArray:
            if not len(staffGradesArray) == self.num_staff:
                raise ValueError("Staff grades array must have the same length as the number of staff in the problem")
            
            self.staff_grades = staffGradesArray
        
        elif staffGradesChoices is not None:
            if len(staffGradesChoices) == 0:
                raise ValueError("Must choose at least one staff grade")
            
            self.staff_grades = []

            for i in range(self.num_staff):
                self.staff_grades.append(grades[random.choice(staffGradesChoices)])


        # get daily quotas, either randomly generate or get from array
        if quotaLimits:

            grade_quotas = [[((random.randrange(quotaLimits[0], quotaLimits[1]) / 100)) for i in range(self.num_days)] for j in range(5)]

            # make dataframe with column names
            # self.daily_quotas = pd.DataFrame(grade_quotas, columns=[self.dates[d].strftime('%d/%m/%Y') for d in range(self.num_days)], index=[key for key in grades.keys()])

            self.daily_quotas = grade_quotas

        elif quotaArray is not None:

            if not len(quotaArray) == 5 or not len(quotaArray[0]) == self.num_days:
                raise ValueError("Quota array must have a length of 5, and each row must have the same length as the number of days in the problem")
            
            self.daily_quotas = quotaArray

        # get staff leave allowance, either randomly generate or get from array
        if leaveAllowanceLimits:
            leave_allowances = [random.randint(leaveAllowanceLimits[0], leaveAllowanceLimits[1]) for i in range(self.num_staff)]

            # self.staff_leave_allowance = pd.DataFrame(leave_allowances, columns=["Leave Allowance (Days)"], index=[f'Employee {x+1}' for x in range(self.num_staff)])
            self.staff_leave_allowance = leave_allowances

        elif leaveAllowanceArray is not None:
            if not len(leaveAllowanceArray) == self.num_staff:
                raise ValueError("Leave allowance array must have the same length as the number of staff in the problem")
            
            self.staff_leave_allowance = leaveAllowanceArray

        # get preference matrix, either randomly generate or get from array
        if preferenceMatrix is None:
            self.preference_matrix = self.generate_preference_matrix(preferencePercentage)

            # self.preference_matrix = pd.DataFrame(preferences, columns=[self.dates[d].strftime('%d/%m/%Y') for d in range(self.num_days)], index=[f'Employee {x+1}' for x in range(self.num_staff)])
        else:    
            preferences = np.array(preferenceMatrix).reshape(self.num_staff, self.num_days)
            if not len(preferences) == self.num_staff or not len(preferences[0]) == self.num_days:
                raise ValueError("Preference matrix must have the same length as the number of staff and days in the problem")
            self.preference_matrix = preferences
            # self.preference_matrix = pd.DataFrame(preferences, columns=[self.dates[d].strftime('%d/%m/%Y') for d in range(self.num_days)], index=[f'Employee {x+1}' for x in range(self.num_staff)])

        # check that all values are defined
        if self.daily_quotas is None:
            raise ValueError("Quotas must be defined")
                
        if self.staff_leave_allowance is None:
            raise ValueError("Staff leave allowance must be defined")
        
        if self.preference_matrix is None:
            raise ValueError("Preference matrix must be defined")
        
        if self.staff_grades is None:
            raise ValueError("Staff grades must be defined")
        
    def __str__(self):
        return f"{self.num_staff} staff, {self.num_days} days,\n grades: {self.staff_grades}\nquotas:\n{self.daily_quotas},\nstaff leave allowance:\n{self.staff_leave_allowance}\npreference matrix:\n{self.preference_matrix} \n past allocation:\n{self.past_allocation}"
    

    # generate a preference matrix if one doesn't exist
    def generate_preference_matrix(self, preferencePercentage):

        # get the number of ones and zeros in the matrix
        num_ones = int(self.num_staff * self.num_days * (preferencePercentage / 100))
        num_zeros = (self.num_staff * self.num_days) - num_ones
        array = np.array([1] * num_ones + [0] * num_zeros)

        # shuffle the array to randomise the order of ones and zeros
        np.random.shuffle(array)
        return array.reshape(self.num_staff,self.num_days)


    # convert the solution to a csv file for download
    def solution_to_csv(self, solution):
        data = []

        data.append([self.num_days])
        data.append([])
        data.append([self.num_staff])
        data.append([])
        data.append(self.staff_grades)
        data.append([])
        data.append(self.calculate_new_leave_allowance(solution))
        data.append([])

        for row in self.daily_quotas:
            data.append(row)
        data.append([])

        for row in solution.to_numpy():
            data.append(row)
        
        return pd.DataFrame(data).to_csv(index=False, header=False)


    # Hard constraints for problem
    def hard_constraints(self):

         # Staff leave allowance not exceeded
        for e in range(self.num_staff):

            if self.past_allocation is not None:
                allocated_days = sum(self.past_allocation[e][d] for d in range(self.num_days))

                self.model.Add(sum(self.l[e, d] for d in range(self.num_days)) <= (self.staff_leave_allowance[e] + allocated_days) )

            else:
                self.model.Add(sum(self.l[e, d] for d in range(self.num_days)) <= self.staff_leave_allowance[e] )
        

        # change percentage quotas to number of staff
        self.percent_to_value()

        # daily leave quota not exceeded
        # for d in range(self.num_days):

        #     if self.past_allocation is not None:
        #         already_allocated = sum(self.past_allocation[e][d] for e in range(self.num_staff))

        
        #     self.model.Add(sum(self.l[e, d] for e in range(self.num_staff)) <= self.daily_quotas_real[self.staff_grades[e]][d])

        # daily leave quota not exceeded
        for d in range(self.num_days):

            for grade in range(5):

                staff_in_grade = [e for e in range(self.num_staff) if self.staff_grades[e] == grade]

                already_allocated = sum(self.past_allocation[e][d] for e in staff_in_grade) if self.past_allocation is not None else 0

                new_allocation = sum(self.l[e, d] for e in staff_in_grade)

                remaining_quota = self.daily_quotas_real[grade][d] - already_allocated

                remaining_quota = max(remaining_quota, already_allocated)

                self.model.Add(new_allocation <= remaining_quota)


            #     self.model.Add(sum(self.l[e, d] for e in range(self.num_staff) if self.staff_grades[e] == grade) <= self.daily_quotas_real[grade][d])

            # already_allocated = sum(self.past_allocation[e][d] for e in range(self.num_staff)) if self.past_allocation is not None else 0
            # total_leave = sum(self.l[e, d] for e in range(self.num_staff))

            # self.model.Add(total_leave - already_allocated <= self.daily_quotas_real[self.staff_grades[e]][d])



        # Don't allocate leave that isnt requested
        for d in range(self.num_days):
            for e in range(self.num_staff):

                if self.past_allocation is not None:
                    if self.past_allocation[e][d] == 1:
                        self.model.Add(self.l[e, d] == 1)
                        continue
                if self.preference_matrix[e][d] == 0:
                    self.model.Add(self.l[e, d] == 0)

        # Max 2 consecutive weeks
        for e in range(self.num_staff):
            for d in range(self.num_days - 10):
                self.model.Add(sum(self.l[e, d + i] for i in range(11)) <= 10)

        # # Honour previous allocation if provided
        # if self.past_allocation is not None:
        #     for e in range(self.num_staff):
        #         for d in range(self.num_days):
        #             if self.past_allocation[e][d] == 1:
        #                 self.model.Add(self.l[e, d] == 1)

    # soft constraints for problem
    def soft_constraints(self):

        # maximise preferences satisfied
        objective_1 = sum(self.preference_matrix[e][d] * self.l[e, d] for e in range(self.num_staff) for d in range(self.num_days))

        # maximise number of consecutive days off
        consecutive_days_off = {}

        for e in range(self.num_staff):
            for d in range(self.num_days-1):
                consecutive_days_off[(e, d)] = self.model.new_bool_var(f"consecutive_days_off_{e}_{d}")

        for e in range(self.num_staff):
            for d in range(self.num_days-1):
                self.model.add_multiplication_equality(consecutive_days_off[(e, d)], [self.l[e, d], self.l[e, d+1]])

        objective_2 = sum(consecutive_days_off[(e, d)] for e in range(self.num_staff) for d in range(self.num_days-1))

        # give preference to days off with higher leave allowance
        objective_3 = sum(self.staff_leave_allowance[e] * self.l[e, d] for e in range(self.num_staff) for d in range(self.num_days))

        # Objective function
        self.model.maximize(
                (self.objective_1_weighting * objective_1) + 
                (self.objective_2_weighting * objective_2) +
                (self.objective_3_weighting * objective_3)
            )   

    # create the model and add variables and constraints
    def createModel(self):

        # create the model
        self.model = cp_model.CpModel()

        # Matrix for leave granted
        # if employee e has day d off, then L[e,d] = 1
        self.l = {}
        for e in range(self.num_staff):
            for d in range(self.num_days):

                self.l[(e, d)] = self.model.new_bool_var(f"L_{e}_d{d}")

             
        # set up hard and soft constraints
        self.hard_constraints()
        self.soft_constraints()

    # solve the model and return the solutions
    def list_all_solutions(self):
        solution_limit = 100
        solution_printer = SolutionPrinter(self.num_staff, self.num_days, self.l, self.preference_matrix, solution_limit, self.startDate, self.staff_leave_allowance, self.daily_quotas_real, self.staff_grades, self.past_allocation)
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.Solve(self.model, solution_printer)
        self.runtime = solver.WallTime()

    # add a past allocation to the model
    def add_past_allocation(self, past_allocation: FileParser, past_start_date: datetime.date):

        # check paramters are compatible
        if not past_allocation.num_staff == self.num_staff:
            raise ValueError("Number of staff in past allocation must be the same as the number of staff in the problem")
        
        if not past_allocation.num_days == self.num_days:
            raise ValueError("Number of days in past allocation must be the same as the number of days in the problem")
        
        # get the intersection of the dates in the past allocation and the current problem
        mask = pd.DataFrame(np.zeros((self.num_staff, self.num_days), dtype=int), columns=[date.strftime('%d/%m/%Y') for date in self.dates])
        previous_allocation = pd.DataFrame(past_allocation.matrix, columns=[date.strftime('%d/%m/%Y') for date in get_dates(past_start_date, past_allocation.num_days)])
        common_cols = mask.columns.intersection(previous_allocation.columns)

        for col in common_cols:
            mask[col] = previous_allocation[col]

        self.past_allocation = mask.to_numpy()

        return
    
    # convert the percentage quotas to the number of staff
    def percent_to_value(self):

        # check if the quotas are already in number of staff
        

        self.daily_quotas_real = self.daily_quotas.copy()

        if self.daily_quotas[0][0] >=1:
            return


        # get the number of staff in each grade
        x, counts = np.unique(self.staff_grades, return_counts=True)
        frequencies = dict(zip(x, counts))

        # change the quotas to the number of staff
        for i in range(5):
            for j in range(self.num_days):
                try:
                    self.daily_quotas_real[i][j] = ceil(self.daily_quotas[i][j] * frequencies[i])
                except KeyError:
                    self.daily_quotas_real[i][j] = 0

        # convert the quotas to integers
        self.daily_quotas_real = np.array(self.daily_quotas_real).astype(int)

    # set the weightings for the objective functions
    def set_weightings(self, objective_1_weighting: int, objective_2_weighting: int, objective_3_weighting: int):
        self.objective_1_weighting = objective_1_weighting
        self.objective_2_weighting = objective_2_weighting
        self.objective_3_weighting = objective_3_weighting

    # set the time limit for the solver
    def set_time_limit(self, time_limit: int):
        self.time_limit = time_limit

    # calculate the new leave allowance for each employee after the allocation
    def calculate_new_leave_allowance(self, solution):

        # get the number of days off for each employee
        leave_days = [sum(solution.iloc[i,:]) for i in range(self.num_staff)]

        # calculate the new leave allowance
        new_leave_allowance = [self.staff_leave_allowance[i] - leave_days[i] for i in range(self.num_staff)]

        return new_leave_allowance
                