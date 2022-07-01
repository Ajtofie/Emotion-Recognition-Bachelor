def input_query(query_statement):
    answer = raw_input("Do you want to " + query_statement + "? [y|n]")
    if answer == "y" or "Y":
        return True
    else:
        return False
