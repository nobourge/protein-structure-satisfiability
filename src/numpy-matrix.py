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
# matrix = [[0 for x in range(sequence_length)] for y in range(sequence_length)]
print("sequence_length", n)
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

# def get_index_matrix(sequence_length, vpool, sol):
#     matrix = numpy.matrix(numpy.zeros(shape=(sequence_length, sequence_length)))
#     # matrix = [[0 for x in range(sequence_length)] for y in range(sequence_length)]
#     print("sequence_length", sequence_length)
#     for index_i in range(sequence_length):
#         for j in range(sequence_length):
#             print("i", index_i)
#             print("j", j)
#
#             location_valued = False
#             for v in range(sequence_length):
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
