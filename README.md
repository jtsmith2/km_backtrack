# km_backtrack

Solves the many-to-many assignment problem.

The method used is the Hungarian algorithm, also known as the Munkres or
Kuhn-Munkres algorithm that has been modified to allow backtracking to 
solve the many-to-many problem.  See [6]

Parameters
----------
cost_matrix : array
The cost matrix of the bipartite graph.

ability_vector : list
An ability limit vector of m agents is La, where La[i] denotes that how many tasks can be 
assigned to agent i at most (0≤i<m)

task_vector : list
A task range vector L is a vector of n tasks, where L[j] denotes that quantity of task j   
must be assigned (0≤j<n).

Returns
-------
row_ind, col_ind : array
An array of row indices and one of corresponding column indices giving
the optimal assignment. The cost of the assignment can be computed
as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
sorted.

Notes
-----
.. versionadded:: 0.1.0

Examples
--------
    >>> c = np.array([[3,0,1,2],[2,3,0,1],[3,0,1,2],[1,0,2,3]])
    >>> La = [2,2,2,2]
    >>> L = [2,2,2,2]
    >>> agents,tasks = kmb(c,La,L)
    >>> zip(agents,tasks)
    array([(0, 3), (0, 0), (1, 2), (1, 3), (2, 1), (2, 2), (3, 0), (3, 1)])
    >>> c[agents,tasks].sum()
    8

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

6. Zhub et al. Solving the Many to Many assignment problem by improving the Kuhn–Munkres algorithm with backtracking
*Theoretical Computer Science*, 618:30-41, March 2016. DOI:http://dx.doi.org/10.1016/j.tcs.2016.01.002
