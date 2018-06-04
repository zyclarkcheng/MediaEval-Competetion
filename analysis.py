if __name__ == '__main__':
     file = open('../data/detect_v0_v25.txt', 'r')

     lines = [line.rstrip('\n') for line in file.readlines()]
     for line in lines:
         columns = line.split(',')

         video = columns[0]
         frame = columns[1]
         category = columns[2]
         count  = columns[3]
         for colum in columns:
             print colum
