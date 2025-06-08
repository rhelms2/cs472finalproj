
def calculateMSE(model, test_data):
    sum_squared_error = 0
    test_print_tally = 0
    for ex in test_data:
        input_params, val = ex
        pred = model.predict(input_params)

        if test_print_tally % 1000 == 0:
            print(f"Prediction: {round(pred, 3)}, actual value in test data: {val}")
            print(f"Sum of squared error at test case {round(test_print_tally, 3)}: {sum_squared_error}")
        test_print_tally += 1

        sum_squared_error += (val - pred)**2

    print("")
    acc = float(sum_squared_error) / len(test_data)
    return acc