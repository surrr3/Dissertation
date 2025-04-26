# File to store classes for the AL Allocation problem

import random
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from math import ceil
import streamlit as st
import datetime

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


# Class to store the previous allocation data
# Used to read and parse previous allocation or new model data from a file
class FileParser():

    num_days = None
    num_staff = None
    leave_allowance = None
    staff_grades = None
    quotas = None
    matrix = None
    start_date = None

    def __init__(self, file):

        self.file = file

    def read_file(self):

        lines = self.file.read().decode("utf-8").splitlines()
        data = [line.split(",") for line in lines]


        self.num_days = int(data[0][0])
        self.num_staff = int(data[2][0])

        self.staff_grades = [int(x) for x in data[4][0:]]

        print(self.staff_grades)
        print(self.num_staff)
        print(self.num_days)

        if len(self.staff_grades) != self.num_staff:
            raise ValueError("Staff grades array must have the same length as the number of staff in the problem")
        
        self.leave_allowance = [int(x) for x in data[6][0:]]

        if len(self.leave_allowance) != self.num_staff:
            raise ValueError("Leave allowance array must have the same length as the number of staff in the problem")
        
        self.quotas = [[float(x) for x in data[8][0:]] for y in range(5)]

        if len(self.quotas) != 5 or len(self.quotas[0]) != self.num_days:
            raise ValueError("Quota array must have a length of 5, and each row must have the same length as the number of days in the problem")
        
        self.matrix = [[int(x) for x in row] for row in data[14:]]

        if len(self.matrix) != self.num_staff or len(self.matrix[0]) != self.num_days:
            raise ValueError("Allocation matrix must have the same length as the number of staff and days in the problem")
        
        


# Function to highlight differences in DataFrame
def highlight_diff(data, other, color='#ff616b'):
    attr = f'background-color: {color}'
    return pd.DataFrame(np.where(data.ne(other), attr, ''),
                        index=data.index, columns=data.columns)

# Function to highlight cells based on value
def highlight_cells(val):
    color = '#1eff00' if val == 1 else ''
    return 'color: %s' % color

def highlight_past_allocation(val):
    color = '#fcc203' if val == 1 else ''
    return 'background-color: %s' % color

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

        # print(f"Hamming Distance: {self.getHammingDistance(solutionArray, self._preference_matrix)}")


        st.session_state.solution_array.append((df_styled,df))
        print(st.session_state.solution_array)

        if self._solution_count >= self._solution_limit:
            print(f"Stop search after {self._solution_limit} solutions")
            self.stop_search()

    # def getMetrics(self, allocation, preferences):

    #     hamming_distance = np.sum(np.abs(np.array(allocation) - np.array(preferences)))
    #     return hamming_distance

    def solutionCount(self):
        return self._solution_count
    
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
    '''
    
    def __init__(self, num_staff: int, num_days: int, quotaLimits: tuple = None, quotaArray: list = None, leaveAllowanceLimits: tuple = None, leaveAllowanceArray: list = None, preferencePercentage:int = None, preferenceMatrix: list = None, startDate: datetime.date = None, staffGradesArray: list = None, staffGradesChoices: list = None ):

        self.num_staff = num_staff
        self.num_days = num_days
        self.startDate = startDate

        self.dates = self.get_dates(self.startDate, self.num_days)

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
        return f"{self.num_staff} staff, {self.num_days} days,\n grades: {self.staff_grades}\nquotas:\n{self.daily_quotas},\nstaff leave allowance:\n{self.staff_leave_allowance}\npreference matrix:\n{self.preference_matrix}"
    

    # Use the start date of the problem to generate a list of dates for display
    def get_dates(self, startDate=None, num_days=None):

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
        

    # generate a preference matrix if one doesn't exist
    def generate_preference_matrix(self, preferencePercentage):

        # get the number of ones and zeros in the matrix
        num_ones = int(self.num_staff * self.num_days * (preferencePercentage / 100))
        num_zeros = (self.num_staff * self.num_days) - num_ones
        array = np.array([True] * num_ones + [False] * num_zeros)

        # shuffle the array to randomise the order of ones and zeros
        np.random.shuffle(array)
        return array.reshape(self.num_staff,self.num_days)

    def dataset_to_csv(self):
        # data = [[self.num_days], [self.num_staff], self.staff_leave_allowance, self.daily_quotas]

        # data.extend(self.preference_matrix.tolist())

        # df = pd.DataFrame(data)
        # csv_data = df.to_csv(index=False, header=False)
        # return csv_data
        return None

    # Hard constraints for problem
    def hard_constraints(self):

         # Staff leave allowance not exceeded
        for e in range(self.num_staff):
            self.model.Add(sum(self.l[e, d] for d in range(self.num_days)) <= self.staff_leave_allowance[e])
        

        # change percentage quotas to number of staff
        self.percent_to_value()

        # daily leave quota not exceeded
        # calculate number of staff * staff_limit for each day
        for d in range(self.num_days):
            self.model.Add(sum(self.l[e, d] for e in range(self.num_staff)) <= self.daily_quotas_real[self.staff_grades[e]][d])

        # Don't allocate leave that isnt requested
        for d in range(self.num_days):
            for e in range(self.num_staff):
                if self.preference_matrix[e][d] == 0:
                    self.model.Add(self.l[e, d] == 0)

        # Max 2 consecutive weeks
        for e in range(self.num_staff):
            for d in range(self.num_days - 10):
                self.model.Add(sum(self.l[e, d + i] for i in range(11)) <= 10)

        # Honour previous allocation if provided
        if self.past_allocation is not None:
            for e in range(self.num_staff):
                for d in range(self.num_days):
                    if self.past_allocation[e][d] == 1:
                        self.model.Add(self.l[e, d] == 1)

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
        solution_printer = SolutionPrinter(self.num_staff, self.num_days, self.l, self.preference_matrix, solution_limit)
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.max_time_in_seconds = 10.0
        solver.Solve(self.model, solution_printer)       

    def add_past_allocation(self, past_allocation: FileParser, past_start_date: datetime.date):

        # check paramters are compatible
        if not past_allocation.num_staff == self.num_staff:
            raise ValueError("Number of staff in past allocation must be the same as the number of staff in the problem")
        
        if not past_allocation.num_days == self.num_days:
            raise ValueError("Number of days in past allocation must be the same as the number of days in the problem")
        
        past_dates = self.get_dates(past_start_date, past_allocation.num_days)

        mask = pd.DataFrame(np.zeros((self.num_staff, self.num_days), dtype=int), columns=[date.strftime('%d/%m/%Y') for date in self.dates])

        previous_allocation = pd.DataFrame(past_allocation.matrix, columns=[date.strftime('%d/%m/%Y') for date in self.get_dates(past_start_date, past_allocation.num_days)])

        common_cols = mask.columns.intersection(previous_allocation.columns)

        for col in common_cols:
            mask[col] = previous_allocation[col]

        self.past_allocation = mask.to_numpy()
        print(self.past_allocation)
        return
    
    # convert the percentage quotas to the number of staff
    def percent_to_value(self):

        self.daily_quotas_real = self.daily_quotas.copy()

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
                