import numpy

n = 2
matrix2 = numpy.matrix(numpy.zeros(shape=(n, n)))

for i in range(n):
    for j in range(n):
        # matrix2[i, j] = 3
        matrix2[i][j] = 3
        print(matrix2)

sol = [6, 0, 1]

matrix = numpy.matrix(numpy.zeros(shape=(n, n)))
# matrix = [[0 for x in range(n)] for y in range(n)]
print("n", n)
for index_i in range(n):
    for j in range(n):
        print("i", index_i)
        print("j", j)

        location_valued = False
        for v in range(n):
            if index_i == 0 :
                print("v: ", v)
                print("type(v): ", type(v))
                matrix[index_i, j] = v
                location_valued = True
        if not location_valued:
            print("Error: no value for location: ", index_i, j)
            matrix[index_i][j] = 333
            # print(matrix)
        print(matrix)

# def get_matrix(n, vpool, sol):
#     matrix = numpy.matrix(numpy.zeros(shape=(n, n)))
#     # matrix = [[0 for x in range(n)] for y in range(n)]
#     print("n", n)
#     for index_i in range(n):
#         for j in range(n):
#             print("i", index_i)
#             print("j", j)
#
#             location_valued = False
#             for v in range(n):
#                 if vpool.id((index_i, j, v)) in sol:
#                     print("v: ", v)
#                     print("type(v): ", type(v))
#                     matrix[index_i, j] = v
#                     location_valued = True
#             if not location_valued:
#                 print("Error: no value for location: ", index_i, j)
#                 matrix[index_i][j] = 333
#                 # print(matrix)
#             print(matrix)
#
#     return matrix
