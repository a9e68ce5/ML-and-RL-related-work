import mdp
import mdp_utils
import numpy as np
import sys
from io import StringIO


# Capture output to a string buffer
output_buffer = StringIO()
sys.stdout = output_buffer

#Create homework grid world 
grid_env = mdp.gen_simple_world()
# Debugging: Print rewards and transitions
print("Rewards array:", grid_env.rewards)
print("Transitions array shape:", grid_env.transitions.shape)

#visualize rewards
print("rewards")
mdp_utils.print_array_as_grid(grid_env.rewards, grid_env)

print("visualize a random policy")
random_pi = mdp_utils.get_random_policy(grid_env)
mdp_utils.visualize_policy(random_pi, grid_env)


##TODO: Implement Value iteration then uncomment lines below 
# print("--- value iteration ---")
# V_vi = mdp_utils.value_iteration(grid_env)
# print("Values from Value Iteration")
# mdp_utils.print_array_as_grid(V_vi, grid_env)
# ##TODO: implement policy extraction from optimal values
# opt_pi = mdp_utils.extract_optimal_policy(V_vi, grid_env)
# print("Optimal Policy")
# mdp_utils.visualize_policy(opt_pi, grid_env)
# --- Implement Value Iteration ---
print("\n--- Value Iteration ---")
V_vi = mdp_utils.value_iteration(grid_env)
print("\nValues from Value Iteration:")
mdp_utils.print_array_as_grid(V_vi, grid_env)

# --- Implement Optimal Policy Extraction ---
print("\nExtracting Optimal Policy:")
opt_pi = mdp_utils.extract_optimal_policy(V_vi, grid_env)
print("\nOptimal Policy:")
mdp_utils.visualize_policy(opt_pi, grid_env)



print("--- policy evaluation ---")
#TODO: Implement Policy Iteration then uncomment the lines below
# policy_pi, V_pi = mdp_utils.policy_iteration(grid_env)
# print("Optimal Policy")
# mdp_utils.visualize_policy(policy_pi, grid_env)
# print("Values from Policy Iteration")
# mdp_utils.print_array_as_grid(V_pi, grid_env)
print("\n--- Policy Iteration ---")
# Implement Policy Iteration
policy_pi, V_pi = mdp_utils.policy_iteration(grid_env)

# Display the optimal policy
print("\nOptimal Policy:")
mdp_utils.visualize_policy(policy_pi, grid_env)

# Display the values obtained from Policy Iteration
print("\nValues from Policy Iteration:")
mdp_utils.print_array_as_grid(V_pi, grid_env)

# Restore standard output
sys.stdout = sys.__stdout__

# Save the captured output to a file
output_results = output_buffer.getvalue()
output_file_path = 'mdp_scenario_results.txt'
with open(output_file_path, 'w') as output_file:
    output_file.write(output_results)

# Output the file path for user access
output_file_path