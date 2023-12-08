import heapq
import sys

def allocate_gpu(tasks, gpus):
    task_sorted = []

    # Create a min heap based on memory_usage using heapq
    min_heap = [(gpu['memory_used'], gpu['index']) for gpu in gpus]
    heapq.heapify(min_heap)

    for task in tasks:
        smallest_memory_usage, gpu = heapq.heappop(min_heap)
        heapq.heappush(min_heap, (smallest_memory_usage + task['gpu_usage'], gpu))
        task['gpu'] = gpu
        del task['estimated_time']
        task_sorted.append(task)

    return task_sorted

def adjusted_execution_time(normal_time, gpu_utilization):
    if gpu_utilization == 0:
        return normal_time  # If GPU utilization is 0%, return normal execution time
    elif gpu_utilization == 100:
        return normal_time / 2  # If GPU utilization is 100%, return half of the normal execution time
    else:
        # For other utilization percentages, calculate the adjusted execution time proportionally
        return normal_time * (0.5 + (gpu_utilization / 200))


def real_min_min(tasks, gpus):
    scheduled_tasks = []
    execution_data = []

    #Create 2D array
    for gpu in gpus:
        temp_list = []
        for task in tasks:
            temp_list.append(adjusted_execution_time(task['estimated_time'], gpu['utilization_gpu']))
        execution_data.append(temp_list)

    for i in range(len(execution_data)):
        min_value = float('inf')  # Initializing with a very large value for comparison
        min_outer_index = -1
        min_inner_index = -1

        # Loop through the outer list
        for outer_index, inner_list in enumerate(execution_data):
            # Loop through the inner list
            for inner_index, value in enumerate(inner_list):
                if value < min_value:
                    min_value = value
                    min_outer_index = outer_index
                    min_inner_index = inner_index

        scheduled_tasks.append(min_outer_index)

        # Looping through the list with its index
        for index, item in enumerate(execution_data):
            if index == min_outer_index:
                item[0] = sys.maxsize
                item[1] = sys.maxsize
            else:
                item[min_inner_index] = item[min_inner_index] + min_value
                
    return scheduled_tasks

def real_max_min(tasks, gpus):
    scheduled_tasks = []
    execution_data = []

    # Create 2D array
    for gpu in gpus:
        temp_list = []
        for task in tasks:
            temp_list.append(adjusted_execution_time(task['estimated_time'], gpu['utilization_gpu']))
        execution_data.append(temp_list)

    for i in range(len(execution_data)):
        max_value = -float('inf')
        max_outer_index = -1
        max_inner_index = -1

        # Loop through the outer list
        for outer_index, inner_list in enumerate(execution_data):
            # Loop through the inner list
            for inner_index, value in enumerate(inner_list):
                if value > max_value:
                    max_value = value
                    max_outer_index = outer_index
                    max_inner_index = inner_index

        scheduled_tasks.append(max_outer_index)

        # Looping through the list with its index
        for index, item in enumerate(execution_data):
            if index == max_outer_index:
                item[0] = -sys.maxsize
                item[1] = -sys.maxsize
            else:
                item[max_inner_index] = item[max_inner_index] - max_value

    return scheduled_tasks


       
    