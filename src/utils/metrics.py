import pandas as pd


def forgetting_table(acc_dataframe, n_tasks=5, column_name="avg_forgetting"):
    """
    Compute forgetting table for every task
    """
    return pd.DataFrame([[forgetting_line(acc_dataframe, task_id=i)] for i in range(n_tasks)], columns=[column_name])

def forgetting_line(acc_dataframe, task_id=4):
    forgettings = [forgetting(task_id, p, acc_dataframe) for p in range(task_id +1)]
    
    # Create dataframe to handle NaN
    return pd.DataFrame(forgettings)

def forgetting(q, p, df):
    D = {}
    for i in range(0, q+1):
        D[f"d{i}"] = df.diff(periods=-i)

    # Create datafrmae to handle NaN
    return pd.DataFrame(([D[f'd{k}'].iloc[q-k,p] for k in range(0, q+1)])).max()[0]