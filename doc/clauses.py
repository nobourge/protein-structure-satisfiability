to_append.append([-neighborhood_symbol
                         , vpool.id((y
                                     , x
                                     , sequence_index1
                                     ))])
to_append.append([-neighborhood_symbol
                     , vpool.id((y2
                                 , x2
                                 , sequence_index2
                                 ))])
to_append.append([-vpool.id((y
                             , x
                             , sequence_index1
                             ))
                     , -vpool.id((y2
                                  , x2
                                  , sequence_index2
                                  ))
                     , neighborhood_symbol])

for index in range(sequence_length):  # take 1 index
    for x in range(matrix_size):
        for y in range(matrix_size):  # take 1 cell
            for x2 in range(matrix_size):
                for y2 in range(matrix_size):  # take 2nd cell
                    if not (x == x2 and
                            y == y2):
                        # cell 1 and 2
                        # can't have same index
                        cnf.append([-vpool.id((x, y, index)),
                                    -vpool.id((x2, y2, index))])
