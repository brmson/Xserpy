import argparse

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Annotate questions with DAGs")
    parser.add_argument("fpath",help="filepath",type=str)
    args = parser.parse_args()