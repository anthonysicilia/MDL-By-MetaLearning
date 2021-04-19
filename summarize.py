import pandas as pd
import sys

if __name__ == '__main__':

    res_file = sys.argv[1]

    df = pd.read_csv(res_file)

    base_name = res_file.split('.')[0]

    df.groupby('model').agg('mean').to_csv(base_name + '-mean.txt')
    df.groupby(['model', 'fold']).agg('std'
        ).groupby('model').agg('mean').to_csv(base_name + '-std.txt')
