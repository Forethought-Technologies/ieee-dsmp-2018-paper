import itertools

def grid_search(method, data, param_options_dict):
  '''
  Iterates through all possible values of param_options_dict, calls the method
  and reports the best parameters.
  '''
  best_accuracy = 0
  best_params = []

  print('Starting grid search!')

  # Iterate over all possible combinations of parameter values
  for fit_values in itertools.product(*param_options_dict.values()):
    # Construct the method arguments as a dictionary of param name and value
    param_dict = dict(zip(param_options_dict.keys(), fit_values))
    
    accuracy = method(data, param_dict)

    print('Tried {}, got '.format(param_dict) + str(accuracy))

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_params = param_dict

  print('Best result of {} was obtained with params'.format(best_accuracy))
  print('Best params:', best_params)
  return best_params
