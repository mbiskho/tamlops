import heapq

def allocate_gpu(tasks, gpus):
    task_sorted = []

    # Create a min heap based on memory_usage using heapq
    min_heap = [(gpu['memory_used'], gpu) for gpu in gpus]
    heapq.heapify(min_heap)

    for task in tasks:
        smallest_memory_usage, gpu = heapq.heappop(min_heap)
        heapq.heappush(min_heap, (smallest_memory_usage + task['gpu_usage'], gpu))
        task['num_gpu'] = gpu['index']
        del task['estimated_time']
        del task['gpu_usage']
        task_sorted.append(task)

    return task_sorted