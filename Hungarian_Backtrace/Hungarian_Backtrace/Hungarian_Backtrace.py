# Hungarian algorithm (Kuhn-Munkres) for solving the linear sum assignment
# problem. Taken from scikit-learn. Based on original code by Brian Clapper,
# adapted to NumPy by Gael Varoquaux.
# Further improvements by Ben Root, Vlad Niculae and Lars Buitinck.
#
# Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
# Author: Brian M. Clapper, Gael Varoquaux
# License: 3-clause BSD

import numpy as np

class KM_Backtrack():

    def kmb(cost_matrix, ability_vector, task_vector):
        """Solve the many-to-many assignment problem.

        The method used is the Hungarian algorithm, also known as the Munkres or
        Kuhn-Munkres algorithm that has been modified to allow backtracking to 
        solve the many-to-many problem.  See:

        Solving the Many to Many assignment problem by improving the Kuhn–Munkres algorithm with backtracking
        Haibin Zhub, Dongning Liua, , , Siqin Zhanga, Yu Zhuc, Luyao Tengd, Shaohua Tenga
        http://dx.doi.org/10.1016/j.tcs.2016.01.002

        Parameters
        ----------
        cost_matrix : array
            The cost matrix of the bipartite graph.

        ability_vector : list
            An ability limit vector of m agents is La, where La[i] denotes that how many tasks can be 
            assigned to agent i at most (0?i<m)

        task_vector : list
            A task range vector L is a vector of n tasks, where L[j] denotes that quantity of task j   
            must be assigned (0?j<n).

        Returns
        -------
        row_ind, col_ind : array
            An array of row indices and one of corresponding column indices giving
            the optimal assignment. The cost of the assignment can be computed
            as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
            sorted; in the case of a square cost matrix they will be equal to
            ``numpy.arange(cost_matrix.shape[0])``.

        Notes
        -----
        .. versionadded:: 0.17.0

        Examples
        --------
        >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        >>> from scipy.optimize import linear_sum_assignment
        >>> row_ind, col_ind = linear_sum_assignment(cost)
        >>> col_ind
        array([1, 0, 2])
        >>> cost[row_ind, col_ind].sum()
        5

        References
        ----------
        1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html

        2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
           *Naval Research Logistics Quarterly*, 2:83-97, 1955.

        3. Harold W. Kuhn. Variants of the Hungarian method for assignment
           problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.

        4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
           *J. SIAM*, 5(1):32-38, March, 1957.

        5. https://en.wikipedia.org/wiki/Hungarian_algorithm
        """
        cost_matrix = np.asarray(cost_matrix)

        if len(cost_matrix.shape) != 2:
            raise ValueError("expected a matrix (2-d array), got a %r array"
                             % (cost_matrix.shape,))

        # The algorithm expects more columns than rows in the cost matrix.
        if cost_matrix.shape[1] < cost_matrix.shape[0]:
            cost_matrix = cost_matrix.T
            transposed = True
        else:
            transposed = False

        state = _Hungary(cost_matrix, task_vector, ability_vector)

        # No need to bother with assignments if one of the dimensions
        # of the cost matrix is zero-length.
        step = None if 0 in cost_matrix.shape else _step1

        while step is not None:
            step = step(state)

        if transposed:
            marked = state.marked.T
        else:
            marked = state.marked
        return np.where(marked == 1)

    # Individual steps of the algorithm follow, as a state machine: they return
    # the next step to be taken (function to be called), if any.

    def _step1(state):
        """Steps 1 and 2 in the Wikipedia page."""

        # Step 1: For each row of the matrix, find the smallest element and
        # subtract it from every element in its row.
        state.C -= state.C.min(axis=1)[:, np.newaxis]
        # Step 2: Find a zero (Z) in the resulting matrix. If there is no
        # starred zero in its row or column, star Z and mark related zeros as unavailable. 
        # Repeat for each element in the matrix.
        for i, j in zip(*np.where(state.C == 0)):
            if state.col_uncovered[j] and state.row_uncovered[i] and state.available[i,j]:
                state._star(i, j)
                state.col_uncovered[j] = False
                state.row_uncovered[i] = False

        state._clear_covers()
        return _step3


    def _step3(state):
        """
        Cover each column containing a starred zero. If n columns are covered,
        the starred zeros describe a complete set of unique assignments.
        In this case, Go to DONE, otherwise, Go to Step 4.
        """
        marked = (state.marked == 1)
        state.col_uncovered[np.any(marked, axis=0)] = False

        if marked.sum() < state.C.shape[0]:
            return _step4


    def _step4(state):
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        # We convert to int as numpy operations are faster on int
        C = (state.C == 0).astype(int)
        covered_C = C * state.row_uncovered[:, np.newaxis]
        covered_C *= np.asarray(state.col_uncovered, dtype=int)
        covered_C *= (state.available).astype(int)
        n = state.C.shape[0]
        m = state.C.shape[1]

        while True:
            # Find an uncovered, available zero
            row, col = np.unravel_index(np.argmax(covered_C), (n, m))
            if covered_C[row, col] == 0:
                return _step6
            else:
                state._prime(row, col)
                # Find the first starred element in the row
                star_col = np.argmax(state.marked[row] == 1)
                if state.marked[row, star_col] != 1:
                    # Could not find one
                    state.Z0_r = row
                    state.Z0_c = col
                    return _step5
                else:
                    col = star_col
                    state.row_uncovered[row] = False
                    state.col_uncovered[col] = True
                    covered_C[:, col] = C[:, col] * (
                        np.asarray(state.row_uncovered, dtype=int)) * (
                        state.available[:, col])
                    covered_C[row] = 0


    def _step5(state):
        """
        Construct a series of alternating primed and starred zeros as follows.
        Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always be one).
        Continue until the series terminates at a primed zero that has no starred
        zero in its column. Unstar each starred zero of the series, star each
        primed zero of the series, erase all primes and uncover every line in the
        matrix. Return to Step 3
        """
        count = 0
        path = state.path
        path[count, 0] = state.Z0_r
        path[count, 1] = state.Z0_c

        while True:
            # Find the first starred element in the col defined by
            # the path.
            row = np.argmax(state.marked[:, path[count, 1]] == 1)
            if state.marked[row, path[count, 1]] != 1:
                # Could not find one
                break
            else:
                count += 1
                path[count, 0] = row
                path[count, 1] = path[count - 1, 1]

            # Find the first prime element in the row defined by the
            # first path step
            col = np.argmax(state.marked[path[count, 0]] == 2)
            if state.marked[row, col] != 2:
                col = -1
            count += 1
            path[count, 0] = path[count - 1, 0]
            path[count, 1] = col

        # Convert paths
        for i in range(count + 1):
            if state.marked[path[i, 0], path[i, 1]] == 1:
                state.marked[path[i, 0], path[i, 1]] = 0
                state._set_related_available(path[i, 0], path[i, 1])
            else:
                state._star(path[i, 0], path[i, 1])

        state._clear_covers()
        # Erase all prime markings
        state.marked[state.marked == 2] = 0
        return _step3


    def _step6(state):
        """
        Add the value found in Step 4 to every element of each covered row,
        and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered lines.
        """
        # the smallest uncovered value in the matrix
        if np.any(state.row_uncovered) and np.any(state.col_uncovered):
            minval = np.min(state.C[state.row_uncovered], axis=0)
            minval = np.min(minval[state.col_uncovered])
            state.C[~state.row_uncovered] += minval
            state.C[:, state.col_uncovered] -= minval
        return _step4

class _Hungary(object):
        """State of the Hungarian algorithm.

        Parameters
        ----------
        cost_matrix : 2D matrix
            The cost matrix. Must have shape[1] >= shape[0].
        """

        def __init__(self, cost_matrix, L, La):
            self.C = cost_matrix.copy()
            self.L = L.copy()
            self.La = La.copy()
            self.n, self.m = self.C.shape

            self._expand_cost_matrix()

            self.k = self.C.shape[0]
            
            self.available = np.ones((k,k), dtype=bool)            
            self.row_uncovered = np.ones(self.k, dtype=bool)
            self.col_uncovered = np.ones(self.k, dtype=bool)
            self.Z0_r = 0
            self.Z0_c = 0
            self.path = np.zeros((2*k, 2), dtype=int)
            self.marked = np.zeros((k, k), dtype=int)

        def _clear_covers(self):
            """Clear all covered matrix cells"""
            self.row_uncovered[:] = True
            self.col_uncovered[:] = True

        def _expand_cost_matrix(self):
            """
            Expands matrix C into a KxK matrix where K is the sum of availble agent slots 
            for tasks (the sum of the elements of La).  It does this in 3 steps:
            1. Expand each row by repeating the columns of each task
            2. Expand each column by repeating the rows of each agent
            3. Filling in with columns of zeros to make the matrix KxK

            Example
            --------------
            L = [1,3]
            La = [1,2,2]
            C = [[2,1],
                 [2,1],
                 [1,2]]

            After Step 1 (first column is repeated once, 2nd column is repeated 3x)
            C = [[2,1,1,1],
                 [2,1,1,1],
                 [1,2,2,2]]

            After Step 2 (1st row is repeated once, 2nd and 3rd rows are repeated twice):
            C = [[2,1,1,1],
                 [2,1,1,1],
                 [2,1,1,1],
                 [1,2,2,2],
                 [1,2,2,2]]

            After Step 3 (a column of zeros is added to make the shape 5x5):
            C = [[2,1,1,1,0],
                 [2,1,1,1,0],
                 [2,1,1,1,0],
                 [1,2,2,2,0],
                 [1,2,2,2,0]]
            """
            self.C = np.repeat(self.C, self.L, axis=1)  #step 1
            self.C = np.repeat(self.C, self.La, axis=0)  #step 2
            zero_cols = np.zeros((self.C.shape[0],self.C.shape[0]-self.C.shape[1]),dtype=int)
            self.C = np.hstack((self.C,zero_cols)) #step 3

            self.agent_row_lookup = range(self.m)
            self.task_column_lookup = range(self.n)

            self.agent_row_lookup = np.repeat(self.agent_row_lookup,self.La)
            self.task_column_lookup = np.repeat(self.task_column_lookup,self.L)

            self.task_columns = {}
            self.agent_rows = {}

            for i,_ in enumerate(self.La): 
                self.agent_rows[i]=np.where(self.agent_row_lookup==i)
            for j,_ in enumerate(self.L):
                self.task_columns[j]=np.where(self.task_column_lookup==j)

        def _set_related_unavailable(self,row,col):
            """Sets 'related' cells (same agent, same task, just on different columns and rows) to 
            unavailable so that the same agent can't be assigned to the same task more than once.

            Parameters:
            row - row of agent
            col - column of task
            """
            agent = agent_row_lookup[row]
            task = task_column_lookup[col]

            related_rows = np.delete(agent_rows[agent],np.where(agent_rows[agent]==row)) #the related rows, excluding the current row
            related_cols = np.delete(task_columns[task],np.where(task_columns[task]==col)) #the related cols, excluding the current col
            for i,j in zip(related_rows,related_cols):
                if self.marked[i,j]!=1:
                    self.available[i,j] = False

        def _make_all_available(self):
            self.available[:,:] = True

        def _set_related_available(self,i,j):
            related_rows = agent_rows[agent] #the related rows
            related_cols = task_columns[task] #the related cols
            self.available[related_rows[0]:related_rows[-1]+1,related_cols[0]:related_cols[-1]+1] = True

        def _star(self,i,j):
            self.marked[i,j]=1
            self._set_related_unavailable(i,j)

        def _prime(self,i,j):
            self.marked[i,j]=2
            self._set_related_unavailable(i,j)



            

