#!/usr/bin/env python3

matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []  # Initialize as an empty list
the_middle = [[row[2], row[3]] for row in matrix]  # Extract 3rd and 4th columns
print("The middle columns of the matrix are: {}".format(the_middle))
